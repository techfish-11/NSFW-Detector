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
from transformers import AutoProcessor, FocalNetForImageClassification, pipeline, AutoModelForSeq2SeqLM
from transformers import AutoTokenizer
from PIL import Image
import torch
from torchvision import transforms
import io
from dotenv import load_dotenv
import asyncio

load_dotenv()

DISCORD_TOKEN = os.getenv('DISCORD_TOKEN')

# インテントの設定
intents = discord.Intents.default()
intents.message_content = True
intents.guilds = True
intents.messages = True
intents.reactions = True  # リアクションに関するインテントを追加

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

# テキスト解析モデルの設定
text_model_name = "eliasalbouzidi/distilbert-nsfw-text-classifier"
text_classifier = pipeline("text-classification", model=text_model_name)

# 翻訳モデルの設定（NLLBモデル: 日本語→英語）
translation_model_name = "facebook/m2m100_1.2B"
translation_tokenizer = AutoTokenizer.from_pretrained(translation_model_name)
translation_model = AutoModelForSeq2SeqLM.from_pretrained(translation_model_name)

class StatusButton(discord.ui.View):
    def __init__(self):
        super().__init__()
        self.add_item(discord.ui.Button(label="Discord BOT Status", url="https://status.sakana-cloud.f5.si/status/techfish"))

@bot.event
async def on_ready():
    print(f'Logged in as {bot.user}')
    await bot.tree.sync()  # Sync commands with Discord

@bot.tree.command(name="help", description="NSFW Detector Bot の使い方を表示します。")
async def nsfw_help(interaction: discord.Interaction):
    embed = discord.Embed(
        title="NSFW Detector Bot 使い方",
        description=("このボットは、画像とテキストがNSFW（Not Safe For Work）かどうかを判断します。\n\n"
                     "画像やテキストを含むメッセージに対して、@NSFW Detector をリプライとして返信してください。\n\n"
                     "このボットは以下のAIモデルを使用して解析します:\n"
                     "1. **MichalMlodawski/nsfw-image-detection-large**: SAFE, QUESTIONABLE, UNSAFE\n"
                     "2. **Falconsai/nsfw_image_detection**: SAFE, NSFW\n"
                     "3. **eliasalbouzidi/distilbert-nsfw-text-classifier**: SAFE, NSFW (テキスト解析)\n\n"
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

            # 画像の解析
            if ref_message.attachments:
                for attachment in ref_message.attachments:
                    if attachment.filename.lower().endswith(('jpg', 'jpeg', 'png')):
                        await analyze_image(attachment, message, analyzing_msg, None)

            # テキストの解析
            if ref_message.content:
                await analyze_text(ref_message.content, message, analyzing_msg)

    # 画像が単独で送られたときの処理
    if message.attachments:
        for i, attachment in enumerate(message.attachments):
            if attachment.filename.lower().endswith(('jpg', 'jpeg', 'png')):
                # 画像の処理を非同期に実行
                await analyze_image(attachment, message, None, i + 1)

async def is_nsfw_image(attachment):
    # 画像をダウンロード
    image_bytes = await attachment.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    # モデル推論を非同期に実行
    label_1, confidence_1 = await asyncio.to_thread(run_model_1, image)
    label_2, confidence_2 = await asyncio.to_thread(run_model_2, image)

    # 高い信頼度のラベルを最終判定に使用
    final_label = 'SAFE'
    if confidence_1 > confidence_2:
        final_label = label_1
    else:
        final_label = label_2

    return final_label == 'UNSAFE'

def run_model_1(image):
    # モデル1による解析
    image_tensor = transform(image).unsqueeze(0)
    inputs = feature_extractor(images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
        probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
        confidence_1, predicted_1 = torch.max(probabilities, 1)
    label_1 = model.config.id2label[predicted_1.item()]
    confidence_1 = confidence_1.item()

    return label_1, confidence_1

def run_model_2(image):
    # モデル2による解析
    result = classifier(image)[0]
    label_2 = result['label']
    confidence_2 = result['score']

    # ラベルの変換
    label_2 = 'SAFE' if label_2 == 'normal' else 'UNSAFE'

    return label_2, confidence_2

async def analyze_image(attachment, message, analyzing_msg, index):
    # 画像をダウンロードして解析
    image_bytes = await attachment.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    # モデル推論を非同期に実行
    label_1, confidence_1 = await asyncio.to_thread(run_model_1, image)
    label_2, confidence_2 = await asyncio.to_thread(run_model_2, image)

    # 高い信頼度のラベルを最終判定に使用
    final_label = 'SAFE'
    if confidence_1 > confidence_2:
        final_label = label_1
    else:
        final_label = label_2

    # メンションされた時だけ埋め込みメッセージの作成
    if not analyzing_msg:
        return

    embed = discord.Embed(
        title=f"🖼️ NSFW コンテンツ判定結果 ({index}) 🖼️",
        description=(f"📄 **ファイル名**: {attachment.filename}\n"
                     f"🔍 **モデル1の判定ラベル**: {label_1} (信頼度: {confidence_1 * 100:.2f}%)\n"
                     f"🔍 **モデル2の判定ラベル**: {label_2} (信頼度: {confidence_2 * 100:.2f}%)\n\n"
                     f"**最終判定結果**: {final_label}\n\n"
                     "この画像の内容を２つのAIで確認しました。上記の結果を参考にしてください。"),
        color=discord.Color.green() if final_label == 'SAFE' else discord.Color.red()
    )

    # 解析中メッセージを上書き
    if analyzing_msg:
        await analyzing_msg.edit(content=None, embed=embed)
    
    # リアクションの追加（メンションされていないときのみ）
    if analyzing_msg:
        await message.add_reaction(f'{index}\u20e3')  # 例: 1️⃣ などのリアクションを追加
        final_reaction = '✅' if final_label == 'SAFE' else '🔞'
        await message.add_reaction(final_reaction)

async def analyze_text(text, message, analyzing_msg):
    # 日本語を英語に翻訳
    translated = await translate_to_english(text)

    # NSFWテキスト解析を非同期に実行
    result = await asyncio.to_thread(text_classifier, translated)
    label = result[0]['label']
    confidence = result[0]['score']

    # メンションされた時だけ埋め込みメッセージの作成
    embed = discord.Embed(
        title="📝 テキスト解析結果 📝",
        description=(f"**翻訳文**: {translated}\n\n"
                     f"**判定ラベル**: {label}\n"
                     f"**信頼度**: {confidence * 100:.2f}%\n\n"
                     "テキストの内容を解析し、最終判定を行いました。"),
        color=discord.Color.green() if label == 'SAFE' else discord.Color.red()
    )

    # 解析中メッセージを上書き
    if analyzing_msg:
        await analyzing_msg.edit(content=None, embed=embed)

    # リアクションの追加（メンションされたときは不要）
    if analyzing_msg:
        final_reaction = '✅' if label == 'SAFE' else '🔞'
        await message.add_reaction(final_reaction)

async def translate_to_english(text):
    inputs = translation_tokenizer(text, return_tensors="pt")
    outputs = translation_model.generate(**inputs)
    translated = translation_tokenizer.decode(outputs[0], skip_special_tokens=True)
    return translated

bot.run(DISCORD_TOKEN)
