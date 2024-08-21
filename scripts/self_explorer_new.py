import argparse
import datetime
import json
import os
import re
import sys
import time

import prompts_new as prompts
from and_controller import list_all_devices, AndroidController
from config import load_config
from model_new import parse_explore_rsp, OpenAIModel, QwenModel
from utils_new import print_with_color, normalized_to_pixel

arg_desc = "AppAgent - Autonomous Exploration"
parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter, description=arg_desc)
parser.add_argument("--app")
parser.add_argument("--root_dir", default="./")
args = vars(parser.parse_args())

configs = load_config()

if configs["MODEL"] == "OpenAI":
    mllm = OpenAIModel(base_url=configs["OPENAI_API_BASE"],
                       api_key=configs["OPENAI_API_KEY"],
                       model=configs["OPENAI_API_MODEL"],
                       temperature=configs["TEMPERATURE"],
                       max_tokens=configs["MAX_TOKENS"])
elif configs["MODEL"] == "Qwen":
    mllm = QwenModel(api_key=configs["DASHSCOPE_API_KEY"],
                     model=configs["QWEN_MODEL"])
else:
    print_with_color(f"ERROR: Unsupported model type {configs['MODEL']}!", "red")
    sys.exit()

app = args["app"]
root_dir = args["root_dir"]

if not app:
    print_with_color("What is the name of the target app?", "blue")
    app = input()
    app = app.replace(" ", "")

work_dir = os.path.join(root_dir, "apps")
if not os.path.exists(work_dir):
    os.mkdir(work_dir)
work_dir = os.path.join(work_dir, app)
if not os.path.exists(work_dir):
    os.mkdir(work_dir)
demo_dir = os.path.join(work_dir, "demos")
if not os.path.exists(demo_dir):
    os.mkdir(demo_dir)
demo_timestamp = int(time.time())
task_name = datetime.datetime.fromtimestamp(demo_timestamp).strftime("self_explore_%Y-%m-%d_%H-%M-%S")
task_dir = os.path.join(demo_dir, task_name)
os.mkdir(task_dir)
docs_dir = os.path.join(work_dir, "auto_docs")
if not os.path.exists(docs_dir):
    os.mkdir(docs_dir)
explore_log_path = os.path.join(task_dir, f"log_explore_{task_name}.txt")
reflect_log_path = os.path.join(task_dir, f"log_reflect_{task_name}.txt")

device_list = list_all_devices()
if not device_list:
    print_with_color("ERROR: No device found!", "red")
    sys.exit()
print_with_color(f"List of devices attached:\n{str(device_list)}", "yellow")
if len(device_list) == 1:
    device = device_list[0]
    print_with_color(f"Device selected: {device}", "yellow")
else:
    print_with_color("Please choose the Android device to start demo by entering its ID:", "blue")
    device = input()
controller = AndroidController(device)
width, height = controller.get_device_size()
if not width and not height:
    print_with_color("ERROR: Invalid device size!", "red")
    sys.exit()
print_with_color(f"Screen resolution of {device}: {width}x{height}", "yellow")

print_with_color("Please enter the description of the task you want me to complete in a few sentences:", "blue")
task_desc = input()

round_count = 0
doc_count = 0
useless_list = set()
last_act = "None"
task_complete = False
while round_count < configs["MAX_ROUNDS"]:
    round_count += 1
    print_with_color(f"Round {round_count}", "yellow")
    controller.get_screenshot(f"{round_count}", task_dir)

    prompt = re.sub(r"<task_description>", task_desc, prompts.self_explore_task_template)
    base64_img = os.path.join(task_dir, f"{round_count}.png")
    print_with_color("Thinking about what to do in the next step...", "yellow")
    status, rsp = mllm.get_model_response(prompt, [base64_img])

    if status:
        with open(explore_log_path, "a") as logfile:
            log_item = {"step": round_count, "prompt": prompt, "image": f"{round_count}.png",
                        "response": rsp}
            logfile.write(json.dumps(log_item) + "\n")
        res = parse_explore_rsp(rsp)
        act_name = res[0]
        if act_name == "FINISH":
            task_complete = True
            break
        if act_name == "tap":
            _, area = res
            normalized_x, normalized_y = map(float, area.split(','))
            x, y = normalized_to_pixel(normalized_x, normalized_y, width, height)
            ret = controller.tap(x, y)
            if ret == "ERROR":
                print_with_color("ERROR: tap execution failed", "red")
                break
        elif act_name == "text":
            _, input_str, display_name = res
            status, rsp = mllm.get_model_response("Guide me to the location of " +  display_name + " within the image by providing its bounding boxes.", [base64_img])
            if status:
                normalized = re.findall(r"\[\[(.*)]]", rsp)[0]
                x_min, y_min, x_max, y_max = map(int, normalized.split(','))
                x_center = (x_min + x_max) / 2
                y_center = (y_min + y_max) / 2
                x, y = normalized_to_pixel(x_center, y_center, width, height)
                ret = controller.tap(x, y)
                if ret == "ERROR":
                    print_with_color("ERROR: tap execution failed", "red")
                    break
            else:
                print_with_color(rsp, "red")
                break
            ret = controller.text(input_str)
            if ret == "ERROR":
                print_with_color("ERROR: text execution failed", "red")
                break
            ret = controller.enter()
            if ret == "ERROR":
                print_with_color("ERROR: enter execution failed", "red")
                break
        # elif act_name == "long_press":
        #     _, area = res
        #     tl, br = elem_list[area - 1].bbox
        #     x, y = (tl[0] + br[0]) // 2, (tl[1] + br[1]) // 2
        #     ret = controller.long_press(x, y)
        #     if ret == "ERROR":
        #         print_with_color("ERROR: long press execution failed", "red")
        #         break
        # elif act_name == "swipe":
        #     _, area, swipe_dir, dist = res
        #     tl, br = elem_list[area - 1].bbox
        #     x, y = (tl[0] + br[0]) // 2, (tl[1] + br[1]) // 2
        #     ret = controller.swipe(x, y, swipe_dir, dist)
        #     if ret == "ERROR":
        #         print_with_color("ERROR: swipe execution failed", "red")
        #         break
        else:
            break
        time.sleep(configs["REQUEST_INTERVAL"])
    else:
        print_with_color(rsp, "red")
        break

if task_complete:
    print_with_color(f"Autonomous exploration completed successfully. {doc_count} docs generated.", "yellow")
elif round_count == configs["MAX_ROUNDS"]:
    print_with_color(f"Autonomous exploration finished due to reaching max rounds. {doc_count} docs generated.",
                     "yellow")
else:
    print_with_color(f"Autonomous exploration finished unexpectedly. {doc_count} docs generated.", "red")
