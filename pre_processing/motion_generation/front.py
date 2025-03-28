from openai import OpenAI

client = OpenAI(
    base_url="https://api.deepseek.com/",
    api_key="sk-eba49ae0531a4619a09bc21a09cc023f"
)

messages = [{"role": "user", "content": "请帮我用Gradio生成一个前端网页，可以让用户从给定物体列表中选择1到3个物体，同时可选是否输入0-2个动作。选择物体的逻辑是鼠标左键点击物体就选中，最多选中3个。在选择完物体和动作后，点击提交按钮，网页会将物体和动作保存为一个JSON文件。JSON文件的格式如下：\n{\n  \"objects\": [\"object1\", \"object2\", \"object3\"],\n  \"action\": \"action\"\n}。得到JSON文件后，使用python处理这个JSON文件，并将处理后的结果显示在前端网页上。"}]

response = client.chat.completions.create(
    model="deepseek-chat",
    messages=messages
)


messages.append(response.choices[0].message)
print(f"Messages Round 1: {messages}")

# messages.append({"role": "user", "content": "在服务器上架设这个网页，通过ip访问，将得到的JSON保存在服务器。"})
# response = client.chat.completions.create(
#     model="deepseek-chat",
#     messages=messages
# )

# messages.append(response.choices[0].message)
# print(f"Messages Round 2: {messages}")