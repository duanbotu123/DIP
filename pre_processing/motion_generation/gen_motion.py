import json
from openai import OpenAI
from typing import Dict, List
import random
import re
import os

object_list = ["小箱子","望远镜","棒球棍","瓶子","毛绒玩具","篮球","显示器", "书","笔记本电脑","手机","键盘","苹果","香蕉","西瓜","辣椒","蘑菇", "盆", "马克杯","浇水壶","茶壶","相机","望远镜","锤子", "橡皮鸭", "小猪存钱罐", "杯子", "大箱子", "中箱子", "旅行箱" ,"椅子", "球形瑜伽球","花生形瑜伽球","垃圾桶","凳子","旅行箱","簸箕","花盆","圆柱", "金字塔玩具"]

client = OpenAI(
    base_url="https://api.deepseek.com/",
    api_key="sk-e0da2d7e50864c3bb8d2db1c91cd4e62"
)

system_prompt = '''
用户将会给你提供1到3个物体的名称,请根据这些物体，设计一个人和这些物体交互的动作，对单物品，设计10-15条不同的动作；对多物体设计10-15条不同的动作。
具体动作内容要求:
1.	有桌和无桌几乎只是物体的区别，但这里的桌不算物体，只是小物体放太低看不清。可以设计有桌物体与无桌物体混合的动作，这个动作视为无桌。比如：坐在椅子上用瓶子喝水。
2.	多物体交互关系不要太复杂，主要怕不好tracking。
3.	动作不要太难或太快。
4.	长度分为两种，简单的和复杂的。简单的3-5秒即可，其中只包含一个完整动作。但语言描述长不一定动作复杂。比如"Lift the floorlamp, move the floorlamp, and put down the floorlamp."看起来是三段，其实可以只算一个，因为拍出来只是拿起来走了一段再放下。总之本质简单即可。
5.	复杂的动作包含多个独立动作，做这个复杂动作需要10秒左右。单物体和多物体都可以设计复杂动作。
6.  禁止出现没有给定的物体。
7.  在你输出的句子中必须出现所有给定的物体，一定要出现物体名称列表中所有的物体，否则非常非常糟糕的事情将会发生。
8.  这些动作中需要混合有桌和无桌,简单和复杂,复杂的动作需要占比30%左右。
9.  需要保证生成的动作符合日常动作，即动作要合理。
10.  对动作的描述仅限于动作本身并且尽可能精确，动作描述中禁止出现人物的性别、年龄、外貌、身份等信息，禁止使用比喻的修辞手法，禁止出现“好像”，“像”。
11.  涉及的物体不要发生非刚性形变，如“弯曲”、“拉伸”、“打开书”、“打开箱子”等。
输入的JSON格式：
{
  "objects": <物体名称列表>,
  "number": <动作条数>
}
输出的JSON格式：
{
    "object_number": <物体数量>,
    "object_list": <物体列表（英文）>,
    "action1":
    {
        "script1": <英文动作描述>,
        "script2": <中文动作描述>,
        "with_desk" : <True or False>,
        "complex": <True or False>
    },
    "action2":
    {
        "script1": <英文动作描述>,
        "script2": <中文动作描述>,
        "with_desk" : <True or False>,
        "complex": <True or False>
    },
    ...
}

EXAMPLE INPUT:
{
    "object": ["盘子", "苹果"],
    "number": 6
}

EXAMPLE JSON OUTPUT:
{   
    "object_number": 2,
    "object_list": ["apple", "plate"],
    "action1":
    {
    "script1": "The person uses his right hand to move the plate,and then he uses his left hand to  place the apple on the plate.",
    "script2": "一个人用他的右手移动盘子，然后用左手将苹果放到盘子上。",
    "with_desk": True,
    "complex": False
    },
    "action2":
    {
    "script1": "The person uses his right hand to get an apple on the floor, then he stands up, taking a bite of the apple, then place the apple on the plate in his left hand.",
    "script2": "一个人用右手从地上拿起苹果，站起来，吃一口苹果，然后将苹果放到左手拿着的盘子上。",
    "with_desk": Flase,
    "complex": True
    },
    ...
    "action6":
    {
    ...
    }
}

EXAMPLE INPUT:
{
    "object": ["杯子"],
    "number": 13
}

EXAMPLE JSON OUTPUT:
{
    "object_number": 1,
    "object_list": ["cup"],
    "action1":
    {
    "script1": "A person holds a cup with his right hand.",
    "script2": "一个人用右手拿着杯子。",
    "with_desk": False,
    "complex": False
    },
    "action2":
    {
    "script1": "The person repeatedly moves the cup left and right with their left hand, picks up the cup with their right hand to drink tea and then placing it back.",
    "script2": "一个人用左手将杯子反复左右移动，然后用右手拿起杯子喝茶，之后把杯子放回去。",
    "with_desk": True,
    "complex": True
    },
    ...
    "action13":
    {
    ...
    }
}
'''

save_folder = "/home/hlp/data/motion_text"

def choose_object():
    classify_number = random.randint(1,100)
    if classify_number <= 20:
        selected_objects = random.sample(object_list, k=3)
    elif classify_number > 20 and classify_number <=45:
        selected_objects = random.sample(object_list, k=2)
    else:
        selected_objects = random.sample(object_list, k=1)
    return selected_objects

def process_selection(selected_objects):
    number = random.randint(10, 15)
    print(number)
    print(selected_objects)
    data = {
        "objects": selected_objects,
        "number": number
    }

    user_prompt = json.dumps(data, ensure_ascii=False, indent=2)

    messages = [{"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}]

    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=messages,
        response_format={
            'type': 'json_object'
        },
    )
    try:
        content = response.choices[0].message.content
        return json.loads(content)
    except json.JSONDecodeError as e:
        print(e)
        return {
            "object_number": 0,
            "object_list": [],
            "action1": {
                "script1": "",
                "script2": ""
            }
        }

def save_response(response: Dict, save_folder: str):
    for key, value in response.items():
        if key == "object_number":
            object_number = value
            continue
        if key == "object_list":
            object_list = value
            objects = "_".join([s.replace(" ", "") for s in object_list])
            print(objects)
            continue
        match = re.search(r"(\d+)", key)
        if match:
            action_number = match.group(0)
        json_name = f'O{object_number}_{objects}_{action_number}.json'
        save_value = {
            "script1":value["script1"],
            "script2":value["script2"]
        }
        os.makedirs(os.path.join(save_folder,str(object_number),objects),exist_ok=True)
        with open(f'{save_folder}/{str(object_number)}/{objects}/{json_name}', 'w') as f:
            json.dump(save_value, f, ensure_ascii=False, indent=2)

def main():
    for i in range(100):
        objects = choose_object()
        responses = process_selection(objects)
        save_response(responses, save_folder)

if __name__ == "__main__":
    main()