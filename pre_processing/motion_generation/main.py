from openai import OpenAI
import json

client = OpenAI(
    base_url="https://api.deepseek.com/",
    api_key="sk-e0da2d7e50864c3bb8d2db1c91cd4e62"
)

system_prompt = '''
用户将会给你提供1到3个物体的名称和可能提供的动词，请根据这些物体和动词，设计五种一个人和这些物体交互的动作，如果提供动作，那么输出的script中必须出现提供的动作，并以JSON格式输出。对一种动作英文输出三个不同的版本但是语义相同，中文输出一个版本。
The user will provide 1 to 3 object names and may also provide verbs. Based on these objects and verbs, design five interactions between a person and these objects. If verbs are provided, they MUST appear in the generated script. For each interaction:
Provide three different English versions with the same meaning.
Provide one Chinese version.
Output the result in JSON format.

EXAMPLE INPUT:
{
    "object": ["篮球", "望远镜"],
    "action": ["拿"]
}

EXAMPLE JSON OUTPUT:
{
    "action1":{
    "object number": 2,
    "object list(English)": ["basketball", "binoculars"],
    "object list(Chinese)": ["篮球", "望远镜"],
    "script1(English)": "A person is holding binoculars in their left hand and a basketball in their right hand.",
    "script2(English)": "A person is carrying binoculars with their left hand and a basketball with their right hand.",
    "script3(English)": "A person is gripping binoculars with their left hand and a basketball with their right hand.",
    "script4(Chinese)": "一个人左手拿着望远镜，右手拿着篮球。"
    },
    "action2":{
    ...
    },
    "action3":{
    ...
    },
    "action4":{
    ...
    },
    "action5":{
    ...
    }
}
'''

motions = {
    "object": ["瑜伽球", "显示器"],
    "action": ["坐", "举"]
}

user_prompt = json.dumps(motions, ensure_ascii=False, indent=2)

messages = [{"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}]

response = client.chat.completions.create(
    model="deepseek-chat",
    messages=messages,
    response_format={
        'type': 'json_object'
    }
)


print(json.loads(response.choices[0].message.content))