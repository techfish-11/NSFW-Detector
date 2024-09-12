# MIT License
# 
# Copyright (c) 2024 TechFish4
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
import os
import discord
from discord.ext import commands
from transformers import AutoProcessor, FocalNetForImageClassification, pipeline
from PIL import Image
import torch
from torchvision import transforms
import io
from dotenv import load_dotenv
import asyncio

load_dotenv()

DISCORD_TOKEN = os.getenv('DISCORD_TOKEN')

intents = discord.Intents.default()
intents.message_content = True
intents.guilds = True
intents.messages = True
intents.reactions = True

bot = commands.Bot(command_prefix='!', intents=intents)

# 画像モデルの設定
model_path = "MichalMlodawski/nsfw-image-detection-large"
feature_extractor = AutoProcessor.from_pretrained(model_path)
model = FocalNetForImageClassification.from_pretrained(model_path)
model.eval()
transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
classifier = pipeline("image-classification", model="Falconsai/nsfw_image_detection")

class StatusButton(discord.ui.View):
    def __init__(self):
        super().__init__()
        self.add_item(discord.ui.Button(label="Discord BOT Status", url="https://status.sakana-cloud.f5.si/status/techfish"))

@bot.event
async def on_ready():
    print(f'Logged in as {bot.user}')
    await bot.tree.sync()

@bot.tree.command(name="help", description="NSFW Detector Bot の使い方を表示します。")
async def nsfw_help(interaction: discord.Interaction):
    embed = discord.Embed(
        title="NSFW Detector Bot 使い方",
        description=("このボットは、画像がNSFW（Not Safe For Work）かどうかを判断します。\n\n"
                     "画像を含むメッセージに対して、@NSFW Detector をリプライとして返信してください。\n\n"
                     "このボットは以下のAIモデルを使用して解析します:\n"
                     "1. **MichalMlodawski/nsfw-image-detection-large**: SAFE, QUESTIONABLE, UNSAFE\n"
                     "2. **Falconsai/nsfw_image_detection**: SAFE, NSFW\n\n"
                     "結果に基づき、最終判定を行います。"),
        color=discord.Color.blue()
    )
    view = StatusButton()
    await interaction.response.send_message(embed=embed, view=view)

@bot.event
async def on_message(message):
    if message.author == bot.user:
        return

    if bot.user in message.mentions:
        if message.reference and message.reference.message_id:
            ref_message = await message.channel.fetch_message(message.reference.message_id)
            analyzing_msg = await message.channel.send("解析中...")

            if ref_message.attachments:
                tasks = [analyze_image(attachment) for attachment in ref_message.attachments if attachment.filename.lower().endswith(('jpg', 'jpeg', 'png'))]
                results = await asyncio.gather(*tasks)

                final_label = 'SAFE'
                description_lines = []
                for idx, result in enumerate(results, start=1):
                    description_lines.append(
                        f"画像 {idx}:\n"
                        f"📄 **ファイル名**: {ref_message.attachments[idx-1].filename}\n"
                        f"🔍 **モデル1の判定ラベル**: {result['model1_label']} (信頼度: {result['model1_confidence'] * 100:.2f}%)\n"
                        f"🔍 **モデル2の判定ラベル**: {result['model2_label']} (信頼度: {result['model2_confidence'] * 100:.2f}%)\n"
                        f"**最終判定結果**: {result['final_label']}\n\n"
                    )

                    if result['final_label'] == 'UNSAFE':
                        final_label = 'UNSAFE'

                embed = discord.Embed(
                    title="🖼️ NSFW コンテンツ判定結果 🖼️",
                    description=(f"画像の内容を確認しました。\n\n"
                                 + "\n".join(description_lines) + "\n\n"
                                 f"**最終判定結果**: {final_label}\n"
                                 "この結果は複数のAIモデルによって確認されたものです。\n\n"
                                 "ご注意: NSFWと判定された画像は、さらに詳細な検査が必要な場合があります。"),
                    color=discord.Color.green() if final_label == 'SAFE' else discord.Color.red()
                )
                await analyzing_msg.edit(content=None, embed=embed)

    if message.attachments:
        for attachment in message.attachments:
            if attachment.filename.lower().endswith(('jpg', 'jpeg', 'png')):
                # 画像が単独で送られたときのリアルタイム解析
                await analyze_image_realtime(attachment, message)

async def analyze_image(attachment):
    image_bytes = await attachment.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    label_1, confidence_1 = await asyncio.to_thread(run_model_1, image)
    label_2, confidence_2 = await asyncio.to_thread(run_model_2, image)

    final_label = label_1 if confidence_1 > confidence_2 else label_2

    return {
        'model1_label': label_1,
        'model1_confidence': confidence_1,
        'model2_label': label_2,
        'model2_confidence': confidence_2,
        'final_label': final_label
    }

async def analyze_image_realtime(attachment, message):
    image_bytes = await attachment.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    label_1, confidence_1 = await asyncio.to_thread(run_model_1, image)
    label_2, confidence_2 = await asyncio.to_thread(run_model_2, image)

    final_label = label_1 if confidence_1 > confidence_2 else label_2

    final_reaction = '✅' if final_label == 'SAFE' else '🔞'
    await message.add_reaction(final_reaction)

def run_model_1(image):
    image_tensor = transform(image).unsqueeze(0)
    inputs = feature_extractor(images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
        probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
        confidence_1, predicted_1 = torch.max(probabilities, 1)
    label_1 = model.config.id2label[predicted_1.item()]
    return label_1, confidence_1.item()

def run_model_2(image):
    result = classifier(image)[0]
    label_2 = 'SAFE' if result['label'] == 'normal' else 'UNSAFE'
    return label_2, result['score']

bot.run(DISCORD_TOKEN)