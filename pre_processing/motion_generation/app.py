import gradio as gr
import json
from openai import OpenAI
from typing import Dict, List

# 预定义的物体列表（可根据需要自行修改）
object_list = ["大箱子", "中箱子", "小箱子", "旅行箱" ,"椅子", "望远镜","球形瑜伽球","花生形瑜伽球","垃圾桶","棒球棍","凳子","水瓶","毛绒玩具","篮球","显示器","旅行箱","书","笔记本电脑","手机","键盘","苹果","香蕉","西瓜","辣椒","蘑菇", "盆", "马克杯","浇水壶","茶壶","相机","望远镜","锤子", "橡皮鸭", "小猪存钱罐","簸箕","花盆","圆柱"]

client = OpenAI(
    base_url="https://api.deepseek.com/",
    api_key="sk-e0da2d7e50864c3bb8d2db1c91cd4e62"
)

system_prompt_gen = '''
用户将会给你提供1到3个物体的名称和可能提供的动词，请根据这些物体和动词，设计五种一个人和这些物体交互的动作，动作尽量复杂一些且动作多样化一些，句子中不要出现副词和形容词。如果提供动作，那么输出的script中必须出现提供的动作。句子中不允许出现未提供的物体。以JSON格式输出。对一种动作英文输出三个不同的版本但是语义相同，中文输出一个版本。输出的JSON需遵守以下的格式:
{
    "action1":{
    "object number": <物体个数(1-3个)>,
    "object list(English)": <物体列表英文>,
    "object list(Chinese)": <物体列表中文>,
    "script1(English)": <英文描述1>,
    "script2(English)": <英文描述2>,
    "script3(English)": <英文描述3>,
    "script4(Chinese)": <中文描述>,
    },
    "action2": <动作信息2>,
    "action3": <动作信息3>,
    "action4": <动作信息4>,
    "action5": <动作信息5>
}
The user will provide 1 to 3 object names and may also provide verbs. Based on these objects and verbs, design five interactions between a person and these objects, the interactions should be a little complex and 5 interactions should be different from each other. Do NOT use adverbs or adjectives in the sentence. If verbs are provided, they MUST appear in the generated script. Objects not provided are NOT allowed to appear in the sentence.
For each interaction:
Provide three different English versions with the same meaning.
Provide one Chinese version.
Output the result in JSON format.

EXAMPLE INPUT:
{
    "object": ["篮球", "望远镜"],
    "action": []
}

EXAMPLE JSON OUTPUT:
{
    "action1":{
    "object number": 2,
    "object list(English)": ["basketball", "binoculars"],
    "object list(Chinese)": ["篮球", "望远镜"],
    "script1": "A person is holding binoculars in their left hand and a basketball in their right hand.",
    "script2": "A person is carrying binoculars with their left hand and a basketball with their right hand.",
    "script3": "A person is gripping binoculars with their left hand and a basketball with their right hand.",
    "script4": "一个人左手拿着望远镜，右手拿着篮球。"
    },
    "action2":{
    "object number": 2,
    "object list(English)": ["basketball", "binoculars"],
    "object list(Chinese)": ["篮球", "望远镜"],
    "script1": "A person is holding binoculars in their left hand while dribbling a basketball with their right hand.",
    "script2": "A person is gripping binoculars in their left hand and bouncing a basketball with their right hand.",
    "script3": "A person is carrying binoculars in their left hand while tapping a basketball with their right hand.",
    "script4": "一个人左手拿着望远镜，右手拍打篮球。"
    },
    "action3":{
    "object number": 2,
    "object list(English)": ["basketball", "binoculars"],
    "object list(Chinese)": ["篮球", "望远镜"],
    "script1": "A person is holding binoculars in their right hand while kicking a basketball with their left foot.",
    "script2": "A person is gripping binoculars with their right hand and striking a basketball with their left foot.",
    "script3": "A person is carrying binoculars in their right hand while using their left foot to kick a basketball.",
    "script4": "一个人右手上拿着望远镜，用左脚踢篮球"
    },
    "action4":{
    "object number": 2,
    "object list(English)": ["basketball", "binoculars"],
    "object list(Chinese)": ["篮球", "望远镜"],
    "script1": "A person is sitting on a basketball while holding binoculars in their hands.",
    "script2": "A person is perched on a basketball with binoculars in their hands.",
    "script3": "A person is seated on a basketball, gripping binoculars.",
    "script4": "一个人坐在篮球上，手上拿着望远镜。"
    },
    "action5":{
    "object number": 2,
    "object list(English)": ["basketball", "binoculars"],
    "object list(Chinese)": ["篮球", "望远镜"],
    "script1": "A person picks up binoculars and basketball from the ground, walks forward, and then puts them down.",
    "script2": "A person retrieves binoculars and a basketball from the floor, moves forward, and sets them down.",
    "script3": "A person lifts binoculars and basketball from the ground, proceeds forward, and places them down.",
    "script4": "一个人从地上拿起望远镜和篮球，往前走然后放下。"
    }
}

EXAMPLE INPUT:
{
    "object": ["凳子"],
    "action": []
}

EXAMPLE JSON OUTPUT:
{
    "action1":{
    "object number": 1,
    "object list(English)": ["chair"],
    "object list(Chinese)": ["凳子"],
    "script1": "A person is sitting on top of a stool.",
    "script2": "Someone is perched on a stool.",
    "script3": "A person is sitting on top of a stool.",
    "script4": "一个人坐在凳子上。"
    },
    "action2":{
    "object number": 1,
    "object list(English)": ["chair"],
    "object list(Chinese)": ["凳子"],
    "script1": "A person is standing on a stool.",
    "script2": "Someone is balancing on top of a stool.",
    "script3": "An individual is upright on a stool.",
    "script4": "一个人站在凳子上"
    },
    "action3":{
    "object number": 1,
    "object list(English)": ["chair"],
    "object list(Chinese)": ["凳子"],
    "script1": "A person is toting a stool with his left hand.",
    "script2": "Someone is carrying a stool with his left hand.",
    "script3": "An individual is gripping a stool with his left hand.",
    "script4": "一个人用左手拿着凳子"
    },
    "action4":{
    "object number": 1,
    "object list(English)": ["chair"],
    "object list(Chinese)": ["凳子"],
    "script1": "A person is righting an overturned stool.",
    "script2": "A person is setting an upturned stool back on its legs.",
    "script3": "A person is putting a tipped-over stool upright.",
    "script4": "一个人把翻倒的凳子扶正"
    },
    "action5":{
    "object number": 1,
    "object list(English)": ["chair"],
    "object list(Chinese)": ["凳子"],
    "script1": "A person is bending over to pick up a stool.",
    "script2": "A person is leaning down to lift a stool.",
    "script3": "A person is stooping to lift a stool.",
    "script4": "一个人弯腰拿起凳子"
    }
}

'''

system_prompt_trans = '''
用户会给你提供一句用于描述人和物体交互动作的文本，文本中会包含一个或多个物体的名称，你需要分析这段文本，按要求提取信息，并将其输出到一个JSON文件中。输出的JSON需要遵循以下格式：
{
    "object number": <物体个数>,
    "object list(English)": <物体列表英文>,
    "object list(Chinese)": <物体列表中文>,
    "script1(English)": <英文描述1>,
    "script2(English)": <英文描述2>,
    "script3(English)": <英文描述3>,
    "script4(Chinese)": <中文原句>
}
The user will provide you with a sentence that describes the interaction between a person and objects, and the text will contain one or more object names. You need to analyze this text, extract the required information as specified, and output it into a JSON file.

EXAMPLE INPUT:
"一个人用左手举起小箱子。"

EXAMPLE JSON OUTPUT:
{
    "object number": 1,
    "object list(English)": ["smallbox"],
    "object list(Chinese)": ["小箱子"],
    "script1(English)": "A person is raising the smallbox using his left hand.",
    "script2(English)": "A person is hoisting the smallbox  using his left hand.",
    "script3(English)": "A person is elevating the smallbox using his left hand.",
    "script4(Chinese)": "一个人用左手举起小箱子。"
}

EXAMPLE INPUT:
一个人左手拿着望远镜，右手拿着篮球。

EXAMPLE JSON OUTPUT:
{
    "object number": 2,
    "object list(English)": ["basketball", "binoculars"],
    "object list(Chinese)": ["篮球", "望远镜"],
    "script1(English)": "A person is holding binoculars in their left hand and a basketball in their right hand.",
    "script2(English)": "A person is carrying binoculars with their left hand and a basketball with their right hand.",
    "script3(English)": "A person is gripping binoculars with their left hand and a basketball with their right hand.",
    "script4(Chinese)": "一个人左手拿着望远镜，右手拿着篮球。"
}

'''



def process_selection(selected_objects, action1, action2):
    # 检查物体选择数量，必须选择 1 到 3 个物体
    if not (1 <= len(selected_objects) <= 3):
        return "请选择1到3个物体！"
    
    # 处理动作输入（0-2个动作），去除空字符
    actions = []
    if action1.strip():
        actions.append(action1.strip())
    if action2.strip():
        actions.append(action2.strip())
    
    # 将多个动作合并为一个字符串（多个动作之间用逗号隔开），如果没有则为空字符串
    action_str = ", ".join(actions) if actions else ""
    
    # 生成 JSON 数据
    data = {
        "objects": selected_objects,
        "action": action_str
    }
    

    # 调用 OpenAI API 处理数据
    response = process_motion(data, system_prompt_gen)
    
    scripts = process_response(response)
    print(scripts)
    return gr.Dropdown(choices=scripts), response

def process_motion(motion, system_prompt):
    '''
    调用 OpenAI API 处理用户输入的物体和动作信息
    Args:
        motion: 用户输入的物体和动作信息
        system_prompt: 系统提示信息
    Returns:
        API response的字典形式
    '''
    user_prompt = json.dumps(motion, ensure_ascii=False, indent=2)

    messages = [{"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}]

    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=messages,
        response_format={
            'type': 'json_object'
        }
    )

    return json.loads(response.choices[0].message.content)

def process_response(response: Dict):
    '''
    处理 API 返回的结果
    Args:
        response: API 返回的结果
    Returns:
        处理后的结果
    '''
    if response:
        # 提取所有动作键（action1-action5）
        action_keys = [f"action{i}" for i in range(1,6) if f"action{i}" in response]
        
        # 生成带序号的选项列表
        scripts = [
            (f"{idx+1}. {response[key]['script4(Chinese)']}", key)
            for idx, key in enumerate(action_keys)
        ]
        return scripts
    else:
        return []
    
def process_custom(custom_motion, system_prompt):
    '''
    调用 OpenAI API 处理用户输入的自定义动作信息
    Args:
        custom_motion: 用户输入的自定义动作信息
        system_prompt: 系统提示信息
    Returns:
        API response的字典形式
    '''
    user_prompt = custom_motion
    messages = [{"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}]
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=messages,
        response_format={
            'type': 'json_object'
        }
    )
    return json.loads(response.choices[0].message.content)


def save_motion(selected_action, custom_action, stored_data):
    # 处理两种情况：选择预设动作或自定义动作
    if custom_action.strip():  # 自定义动作处理
        custom_data = process_custom(custom_action, system_prompt_trans)
        with open('data.json', 'w', encoding='utf-8') as json_file:
            json.dump(custom_data, json_file, ensure_ascii=False, indent=4)
        return {
            "status": "自定义动作保存成功",
            "data": custom_data
        }
    else:  # 预设动作处理
        # 从存储数据中查找对应动作
        action_number = int(selected_action.replace("action", ""))
        action_data = stored_data.get(selected_action)
        with open('data.json', 'w', encoding='utf-8') as json_file:
            json.dump(action_data, json_file, ensure_ascii=False, indent=4)
        return {
            "status": "预设动作保存成功",
            "data": action_data
        }

with gr.Blocks() as demo:
    gr.Markdown("## 请输入你的名字")
    hidden_data = gr.JSON(visible=False)

    with gr.Row():
        user_name = gr.Textbox(label="名字/昵称")
    
    gr.Markdown("## 请选择物体与输入动作")
    
    with gr.Row():
        object_selector = gr.CheckboxGroup(label="请选择1到3个物体", choices=object_list)
    
    with gr.Row():
        action1 = gr.Textbox(label="动作1（可选）", placeholder="输入动作1")
        action2 = gr.Textbox(label="动作2（可选）", placeholder="输入动作2")

    submit_btn1 = gr.Button("Run")

    gr.Markdown("## 请选择你的动作")
    with gr.Row():
        motion_selector = gr.Dropdown(choices=[], label="请选择动作", info="如果没有你想做的动作，请填写下方的自定义动作",interactive=True, allow_custom_value=True)
    submit_btn1.click(fn=process_selection, inputs=[object_selector, action1, action2], outputs=[motion_selector, hidden_data])

    with gr.Row():
        motion_script = gr.Textbox(label="自定义动作", placeholder="自定义动作")
    
    submit_btn2 = gr.Button("确认")
    submit_btn2.click(
    fn=save_motion,
    inputs=[motion_selector, motion_script, hidden_data],  # 接收三个输入
    outputs=[gr.Textbox(label="保存状态")]  # 添加状态显示组件
    )

demo.launch(share=True, show_error=True)