import os
import base64
import io
import json
import torch
import numpy as np
from PIL import Image
import requests
from io import BytesIO
from google import genai
from google.genai import types
import traceback

class GeminiImageGenerator:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True}),
                "api_key": ("STRING", {"default": "", "multiline": False}),
                "model": (["models/gemini-2.0-flash-preview-image-generation","models/gemini-2.0-flash-exp"], {"default": "gemini-2.0-flash-preview-image-generation"}),
                "aspect_ratio": ([
                    "Free (自由比例)",
                    "Landscape (横屏)",
                    "Portrait (竖屏)",
                    "Square (方形)",
                ], {"default": "Free (自由比例)"}),
                "temperature": ("FLOAT", {"default": 1, "min": 0.0, "max": 2.0, "step": 0.05}),
            },
            "optional": {
                "seed": ("INT", {"default": 66666666, "min": 0, "max": 2147483647}),
                "images": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "API Respond")
    FUNCTION = "generate_image"
    CATEGORY = "Google-Gemini"
    
    def __init__(self):
        """初始化日志系统和API密钥存储"""
        self.log_messages = []  # 全局日志消息存储
        # 获取节点所在目录
        self.node_dir = os.path.dirname(os.path.abspath(__file__))
        self.key_file = os.path.join(self.node_dir, "gemini_api_key.txt")
        
        # 检查google-genai版本
        try:
            import importlib.metadata
            genai_version = importlib.metadata.version('google-genai')
            self.log(f"当前google-genai版本: {genai_version}")
            
            # 检查PIL/Pillow版本
            try:
                import PIL
                self.log(f"当前PIL/Pillow版本: {PIL.__version__}")
            except Exception as e:
                self.log(f"无法检查PIL/Pillow版本: {str(e)}")
            
            # 检查版本是否满足最低要求
            from packaging import version
            if version.parse(genai_version) < version.parse('1.5.0'):  
                self.log("警告: google-genai版本过低，建议升级到最新版本")
                self.log("建议执行: pip install -q -U google-genai")
            
            # 检查PIL/Pillow版本是否满足要求
            try:
                if version.parse(PIL.__version__) < version.parse('9.5.0'):
                    self.log("警告: PIL/Pillow版本过低，建议升级到9.5.0或更高版本")
                    self.log("建议执行: pip install -U Pillow>=9.5.0")
            except Exception:
                pass
        except Exception as e:
            self.log(f"无法检查版本信息: {e}")
    
    def log(self, message):
        """全局日志函数：记录到日志列表"""
        if hasattr(self, 'log_messages'):
            self.log_messages.append(message)
        return message
    
    def get_api_key(self, user_input_key):
        """获取API密钥，优先使用用户输入的密钥"""
        # 如果用户输入了有效的密钥，使用并保存
        if user_input_key and len(user_input_key) > 10:
            self.log("使用用户输入的API密钥")
            # 保存到文件中
            try:
                with open(self.key_file, "w") as f:
                    f.write(user_input_key)
                self.log("已保存API密钥到节点目录")
            except Exception as e:
                self.log(f"保存API密钥失败: {e}")
            return user_input_key
            
        # 如果用户没有输入，尝试从文件读取
        if os.path.exists(self.key_file):
            try:
                with open(self.key_file, "r") as f:
                    saved_key = f.read().strip()
                if saved_key and len(saved_key) > 10:
                    self.log("使用已保存的API密钥")
                    return saved_key
            except Exception as e:
                self.log(f"读取保存的API密钥失败: {e}")
                
        # 如果都没有，返回空字符串
        self.log("警告: 未提供有效的API密钥")
        return ""
    
    def generate_empty_image(self, width=512, height=512):
        """生成标准格式的空白RGB图像张量 - 使用默认尺寸"""
        # 根据比例设置默认尺寸
        empty_image = np.ones((height, width, 3), dtype=np.float32) * 0.2
        tensor = torch.from_numpy(empty_image).unsqueeze(0) # [1, H, W, 3]
        
        self.log(f"创建ComfyUI兼容的空白图像: 形状={tensor.shape}, 类型={tensor.dtype}")
        return tensor
    
    def validate_and_fix_tensor(self, tensor, name="图像"):
        """验证并修复张量格式，确保完全兼容ComfyUI"""
        try:
            # 基本形状检查
            if tensor is None:
                self.log(f"警告: {name} 是None")
                return None
                
            self.log(f"验证 {name}: 形状={tensor.shape}, 类型={tensor.dtype}, 设备={tensor.device}")
            
            # 确保形状正确: [B, C, H, W]
            if len(tensor.shape) != 4:
                self.log(f"错误: {name} 形状不正确: {tensor.shape}")
                return None
                
            if tensor.shape[1] != 3:
                self.log(f"错误: {name} 通道数不是3: {tensor.shape[1]}")
                return None
                
            # 确保类型为float32
            if tensor.dtype != torch.float32:
                self.log(f"修正 {name} 类型: {tensor.dtype} -> torch.float32")
                tensor = tensor.to(dtype=torch.float32)
                
            # 确保内存连续
            if not tensor.is_contiguous():
                self.log(f"修正 {name} 内存布局: 使其连续")
                tensor = tensor.contiguous()
                
            # 确保值范围在0-1之间
            min_val = tensor.min().item()
            max_val = tensor.max().item()
            
            if min_val < 0 or max_val > 1:
                self.log(f"修正 {name} 值范围: [{min_val}, {max_val}] -> [0, 1]")
                tensor = torch.clamp(tensor, 0.0, 1.0)
                
            return tensor
        except Exception as e:
            self.log(f"验证张量时出错: {e}")
            traceback.print_exc()
            return None
    
    def generate_image(self, prompt, api_key, model, aspect_ratio, temperature, seed=66666666, images=None):
        """生成图像 - 支持多张参考图片"""
        response_text = ""
        
        # 重置日志消息
        self.log_messages = []
        
        try:
            # 获取API密钥
            actual_api_key = self.get_api_key(api_key)
            
            if not actual_api_key:
                error_message = "错误: 未提供有效的API密钥。请在节点中输入API密钥或确保已保存密钥。"
                self.log(error_message)
                full_text = "## 错误\n" + error_message + "\n\n## 使用说明\n1. 在节点中输入您的Google API密钥\n2. 密钥将自动保存到节点目录，下次可以不必输入"
                return (self.generate_empty_image(512, 512), full_text)  # 使用默认尺寸的空白图像
            
            # 创建客户端实例
            client = genai.Client(api_key=actual_api_key)
            
            # 处理种子值
            if seed == 0:
                import random
                seed = random.randint(1, 2**31 - 1)
                self.log(f"生成随机种子值: {seed}")
            else:
                self.log(f"使用指定的种子值: {seed}")
            
            # 直接从选择确定方向和提示词
            if "Free" in aspect_ratio:
                # 不指定任何方向或比例限制
                simple_prompt = f"Create a detailed image of: {prompt}."
            elif "Landscape" in aspect_ratio:
                orientation = "wide rectangular image where width is greater than height"
                simple_prompt = f"Generate the image as a {orientation}.Create a detailed image of: {prompt}."
            elif "Portrait" in aspect_ratio:
                orientation = "tall rectangular image where height is greater than width"
                simple_prompt = f"Generate the image as a {orientation}.Create a detailed image of: {prompt}."
            else:  # Square
                orientation = "square image where width equals height"
                simple_prompt = f"Generate the image as a {orientation}.Create a detailed image of: {prompt}."
            
            # 配置生成参数，使用用户指定的温度值
            gen_config = types.GenerateContentConfig(
                temperature=temperature,
                seed=seed,
                response_modalities=['Text', 'Image']
            )
            
            # 记录温度设置
            self.log(f"使用温度值: {temperature}，种子值: {seed}")
            
            # 处理参考图像 - 修改为处理多张图片
            contents = []
            reference_images_count = 0
            
            # 首先添加文本提示
            text_part = {"text": simple_prompt}
            contents.append(text_part)
            
            # 处理参考图像(单张或多张)
            if images is not None:
                try:
                    # 确定图像数量
                    batch_size = images.shape[0]
                    self.log(f"检测到 {batch_size} 张参考图像")
                    
                    # 逐一处理每张图像
                    for i in range(batch_size):
                        # 获取单张图像
                        input_image = images[i].cpu().numpy()
                        
                        # 转换为PIL图像
                        input_image = (input_image * 255).astype(np.uint8)
                        pil_image = Image.fromarray(input_image)
                        
                        self.log(f"参考图像 {i+1} 处理成功，尺寸: {pil_image.width}x{pil_image.height}")
                        
                        # 直接在内存中处理，不保存为文件
                        img_byte_arr = BytesIO()
                        pil_image.save(img_byte_arr, format='PNG')
                        img_byte_arr.seek(0)
                        image_bytes = img_byte_arr.read()
                        
                        # 添加图像部分
                        img_part = {"inline_data": {"mime_type": "image/png", "data": image_bytes}}
                        contents.append(img_part)
                        reference_images_count += 1
                    
                    # 如果有参考图像，添加说明
                    if reference_images_count > 0:
                        if reference_images_count == 1:
                            contents[0]["text"] += " Use this reference image as guidance."
                        else:
                            contents[0]["text"] += f" Use these {reference_images_count} reference images as guidance."
                    
                    self.log(f"成功添加 {reference_images_count} 张参考图像到请求中")
                except Exception as img_error:
                    self.log(f"参考图像处理错误: {str(img_error)}")
                    # 如果处理图像出错，只使用文本提示
                    contents = [{"text": simple_prompt}]
            else:
                # 没有参考图像，只使用文本
                contents = [{"text": simple_prompt}]
            
            # 打印请求信息
            self.log(f"请求Gemini API生成图像，种子值: {seed}, 包含参考图像数: {reference_images_count}")
            
            # 调用API
            response = client.models.generate_content(
                model="models/gemini-2.0-flash-exp",
                contents=contents,
                config=gen_config
            )
            
            # 响应处理
            self.log("API响应接收成功，正在处理...")
            
            if not hasattr(response, 'candidates') or not response.candidates:
                self.log("API响应中没有candidates")
                # 合并日志和返回值
                full_text = "\n".join(self.log_messages) + "\n\nAPI返回了空响应"
                return (self.generate_empty_image(512, 512), full_text)
            
            # 检查响应中是否有图像
            image_found = False
            
            # 遍历响应部分
            for part in response.candidates[0].content.parts:
                # 检查是否为文本部分
                if hasattr(part, 'text') and part.text is not None:
                    text_content = part.text
                    response_text += text_content
                    self.log(f"API返回文本: {text_content[:100]}..." if len(text_content) > 100 else text_content)
                
                # 检查是否为图像部分
                elif hasattr(part, 'inline_data') and part.inline_data is not None:
                    self.log("API返回数据解析处理")
                    try:
                        # 获取图像数据
                        image_data = part.inline_data.data
                        mime_type = part.inline_data.mime_type if hasattr(part.inline_data, 'mime_type') else "未知"
                        self.log(f"图像数据类型: {type(image_data)}, MIME类型: {mime_type}, 数据长度: {len(image_data) if image_data else 0}")
                        
                        # 记录前8个字节用于诊断
                        if image_data and len(image_data) > 8:
                            hex_prefix = ' '.join([f'{b:02x}' for b in image_data[:8]])
                            self.log(f"图像数据前8字节: {hex_prefix}")
                            
                            # 检测Base64编码的PNG
                            if hex_prefix.startswith('69 56 42 4f 52'):
                                try:
                                    self.log("检测到Base64编码的PNG，正在解码...")
                                    base64_str = image_data.decode('utf-8', errors='ignore')
                                    image_data = base64.b64decode(base64_str)
                                    self.log(f"Base64解码成功，新数据长度: {len(image_data)}")
                                except Exception as e:
                                    self.log(f"Base64解码失败: {str(e)}")
                        
                        # BytesIO正确使用方法 - 修改为更直接的初始化方式
                        try:
                            # 直接使用字节数据初始化BytesIO，更简洁更兼容
                            buffer = BytesIO(image_data)
                            
                            # 尝试打开图像
                            pil_image = Image.open(buffer)
                            self.log(f"成功打开图像: {pil_image.width}x{pil_image.height}, 格式: {pil_image.format}")
                            
                            # 确保是RGB模式
                            if pil_image.mode != 'RGB':
                                pil_image = pil_image.convert('RGB')
                            
                            # 不再调整大小，直接使用API返回的尺寸
                            # 删除之前的尺寸调整代码块

                            # 转换为ComfyUI格式
                            img_array = np.array(pil_image).astype(np.float32) / 255.0
                            img_tensor = torch.from_numpy(img_array).unsqueeze(0)

                            self.log(f"图像转换为张量成功, 形状: {img_tensor.shape}")
                            
                            # 合并日志和API返回文本
                            full_text = "## 处理日志\n" + "\n".join(self.log_messages) + "\n\n## API返回\n" + response_text
                            return (img_tensor, full_text)
                        
                        except Exception as e:
                            self.log(f"使用BytesIO打开图像失败: {str(e)}")
                            self.log("无法处理图像数据，使用默认空白图像")
                            img_tensor = self.generate_empty_image(512, 512)
                    except Exception as e:
                        self.log(f"图像处理错误: {e}")
                        traceback.print_exc()  # 添加详细的错误追踪信息
            
            # 没有找到图像数据，但可能有文本
            if not image_found:
                self.log("API响应中未找到图像数据，仅返回文本")
                if not response_text:
                    response_text = "API未返回任何图像或文本"
            
            # 合并日志和API返回文本
            full_text = "## 处理日志\n" + "\n".join(self.log_messages) + "\n\n## API返回\n" + response_text
            return (self.generate_empty_image(512, 512), full_text)
        
        except Exception as e:
            error_message = f"处理过程中出错: {str(e)}"
            self.log(f"Gemini图像生成错误: {str(e)}")
            
            # 合并日志和错误信息
            full_text = "## 处理日志\n" + "\n".join(self.log_messages) + "\n\n## 错误\n" + error_message
            return (self.generate_empty_image(512, 512), full_text)

class GeminiMultiImageGenerator(GeminiImageGenerator):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True}),
                "api_key": ("STRING", {"default": "", "multiline": False}),
                "model": (["models/gemini-2.0-flash-exp"], {"default": "models/gemini-2.0-flash-exp"}),
                "width": ("INT", {"default": 1024, "min": 512, "max": 2048, "step": 8}),
                "height": ("INT", {"default": 1024, "min": 512, "max": 2048, "step": 8}),
                "temperature": ("FLOAT", {"default": 1, "min": 0.0, "max": 2.0, "step": 0.05}),
            },
            "optional": {
                "seed": ("INT", {"default": 66666666, "min": 0, "max": 2147483647}),
                "image1": ("IMAGE",),
                "image2": ("IMAGE",),
                "image3": ("IMAGE",),
                "image4": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "API Respond")
    FUNCTION = "generate_multi_image"
    CATEGORY = "Google-Gemini"

    def process_single_image(self, image, idx):
        """处理单张图片"""
        try:
            if len(image.shape) == 4 and image.shape[0] == 1:
                input_image = image[0].cpu().numpy()
                input_image = (input_image * 255).astype(np.uint8)
                pil_image = Image.fromarray(input_image)
                
                img_byte_arr = BytesIO()
                pil_image.save(img_byte_arr, format='PNG')
                img_byte_arr.seek(0)
                image_bytes = img_byte_arr.read()
                
                self.log(f"参考图片 {idx} 处理成功，尺寸: {pil_image.width}x{pil_image.height}")
                return {"mime_type": "image/png", "data": image_bytes}
        except Exception as e:
            self.log(f"处理参考图片 {idx} 失败: {str(e)}")
        return None

    def generate_multi_image(self, prompt, api_key, model, width, height, temperature, seed=66666666, **kwargs):
        """支持多张参考图片的生成函数"""
        response_text = ""
        self.log_messages = []
        
        try:
            actual_api_key = self.get_api_key(api_key)
            if not actual_api_key:
                error_message = "错误: 未提供有效的API密钥。请在节点中输入API密钥或确保已保存密钥。"
                self.log(error_message)
                full_text = "## 错误\n" + error_message + "\n\n## 使用说明\n1. 在节点中输入您的Google API密钥\n2. 密钥将自动保存到节点目录，下次可以不必输入"
                return (self.generate_empty_image(512, 512), full_text)  # 使用默认尺寸的空白图像

            client = genai.Client(api_key=actual_api_key)
            
            # 处理种子值
            if seed == 0:
                import random
                seed = random.randint(1, 2**31 - 1)
                self.log(f"生成随机种子值: {seed}")
            else:
                self.log(f"使用指定的种子值: {seed}")
            
            # 构建提示词
            aspect_ratio = width / height
            orientation = "landscape (horizontal)" if aspect_ratio > 1 else "portrait (vertical)" if aspect_ratio < 1 else "square"
            simple_prompt = f"Create a detailed image of: {prompt}. Generate the image in {orientation} orientation with exact dimensions of {width}x{height} pixels."

            # 处理所有参考图片
            contents = []
            ref_count = 0
            
            # 处理所有图片输入
            for key, image in kwargs.items():
                if key.startswith('image') and image is not None:
                    img_data = self.process_single_image(image, ref_count + 1)
                    if img_data:
                        contents.append({"inline_data": img_data})
                        ref_count += 1
                        self.log(f"参考图片 {key} 处理成功")

            # 添加提示词
            if ref_count > 0:
                contents.append({"text": f"{simple_prompt} Use these {ref_count} reference images as style guidance."})
                self.log(f"成功添加 {ref_count} 张参考图片")
            else:
                contents = simple_prompt

            # 生成配置
            gen_config = types.GenerateContentConfig(
                temperature=temperature,
                seed=seed,
                response_modalities=['Text', 'Image']
            )

            # 调用API
            self.log(f"请求Gemini API生成图像，种子值: {seed}, 包含参考图片数: {ref_count}")
            response = client.models.generate_content(
                model=model,
                contents=contents,
                config=gen_config
            )

            # 响应处理
            self.log("API响应接收成功，正在处理...")
            
            if not hasattr(response, 'candidates') or not response.candidates:
                self.log("API响应中没有candidates")
                full_text = "\n".join(self.log_messages) + "\n\nAPI返回了空响应"
                return (self.generate_empty_image(width, height), full_text)
            
            # 检查响应中是否有图像
            image_found = False
            
            # 遍历响应部分
            for part in response.candidates[0].content.parts:
                # 检查是否为文本部分
                if hasattr(part, 'text') and part.text is not None:
                    text_content = part.text
                    response_text += text_content
                    self.log(f"API返回文本: {text_content[:100]}..." if len(text_content) > 100 else text_content)
                
                # 检查是否为图像部分
                elif hasattr(part, 'inline_data') and part.inline_data is not None:
                    self.log("API返回数据解析处理")
                    try:
                        # 获取图像数据
                        image_data = part.inline_data.data
                        mime_type = part.inline_data.mime_type if hasattr(part.inline_data, 'mime_type') else "未知"
                        self.log(f"图像数据类型: {type(image_data)}, MIME类型: {mime_type}, 数据长度: {len(image_data) if image_data else 0}")
                        
                        # 记录前8个字节用于诊断
                        if image_data and len(image_data) > 8:
                            hex_prefix = ' '.join([f'{b:02x}' for b in image_data[:8]])
                            self.log(f"图像数据前8字节: {hex_prefix}")
                            
                            # 检测Base64编码的PNG
                            if hex_prefix.startswith('69 56 42 4f 52'):
                                try:
                                    self.log("检测到Base64编码的PNG，正在解码...")
                                    base64_str = image_data.decode('utf-8', errors='ignore')
                                    image_data = base64.b64decode(base64_str)
                                    self.log(f"Base64解码成功，新数据长度: {len(image_data)}")
                                except Exception as e:
                                    self.log(f"Base64解码失败: {str(e)}")
                        
                        # BytesIO正确使用方法 - 修改为更直接的初始化方式
                        try:
                            # 直接使用字节数据初始化BytesIO，更简洁更兼容
                            buffer = BytesIO(image_data)
                            
                            # 尝试打开图像
                            pil_image = Image.open(buffer)
                            self.log(f"成功打开图像: {pil_image.width}x{pil_image.height}, 格式: {pil_image.format}")
                            
                            if pil_image.mode != 'RGB':
                                pil_image = pil_image.convert('RGB')
                            
                            # 转换为ComfyUI格式
                            img_array = np.array(pil_image).astype(np.float32) / 255.0
                            img_tensor = torch.from_numpy(img_array).unsqueeze(0)
                            
                            image_found = True
                            self.log(f"图像转换为张量成功, 形状: {img_tensor.shape}")
                            
                            # 合并日志和API返回文本
                            full_text = "## 处理日志\n" + "\n".join(self.log_messages) + "\n\n## API返回\n" + response_text
                            return (img_tensor, full_text)
                            
                        except Exception as e:
                            self.log(f"使用BytesIO打开图像失败: {str(e)}")
                            self.log("无法处理图像数据，使用默认空白图像")
                            img_tensor = self.generate_empty_image(width, height)
                    except Exception as e:
                        self.log(f"图像处理错误: {e}")
                        traceback.print_exc()
            
            # 没有找到图像数据，但可能有文本
            if not image_found:
                self.log("API响应中未找到图像数据，仅返回文本")
                if not response_text:
                    response_text = "API未返回任何图像或文本"
            
            # 合并日志和API返回文本
            full_text = "## 处理日志\n" + "\n".join(self.log_messages) + "\n\n## API返回\n" + response_text
            return (self.generate_empty_image(width, height), full_text)

        except Exception as e:
            error_message = f"处理过程中出错: {str(e)}"
            self.log(f"Gemini多图片生成错误: {str(e)}")
            full_text = "## 处理日志\n" + "\n".join(self.log_messages) + "\n\n## 错误\n" + error_message
            return (self.generate_empty_image(width, height), full_text)

# 注册节点
NODE_CLASS_MAPPINGS = {
    "Google-Gemini": GeminiImageGenerator,
    "Google-Gemini-Multi": GeminiMultiImageGenerator
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Google-Gemini": "Gemini 2.0 image",
    "Google-Gemini-Multi": "Gemini 2.0 Multi-Image"
}