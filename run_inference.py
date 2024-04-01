# Copyright (c) Alibaba, Inc. and its affiliates.
import json
import os

import cv2

from project_env import init_env

init_env()

from facechain.constants import neg_prompt, pos_prompt_with_cloth, pos_prompt_with_style, base_models
from facechain.inference import GenPortrait
from facechain.utils import snapshot_download_dk, PROJECT_DIR
from facechain.wktk.base_utils import PF, TimeMarker, DateTime


def generate_pos_prompt(
        style_model,
        prompt_cloth,
        styles,
        user_prompt_cloth='',
        user_prompt_style=''
):
    pos_prompt_with_cloth2 = pos_prompt_with_cloth
    if user_prompt_cloth is not None and len(user_prompt_cloth) > 4:
        pos_prompt_with_cloth2 = user_prompt_cloth

    pos_prompt_with_style2 = pos_prompt_with_style
    if user_prompt_style is not None and len(user_prompt_style) > 4:
        pos_prompt_with_style2 = user_prompt_style

    if style_model is not None:
        hited_list = list(filter(lambda style: style_model == style['name'], styles))
        if len(hited_list) == 0:
            raise ValueError(f'styles not found: {style_model}')
        hited = hited_list[0]
        if hited['model_id'] is None:
            pos_prompt = pos_prompt_with_cloth2.replace('__prompt_cloth__', prompt_cloth)
        else:
            pos_prompt = pos_prompt_with_style2.replace('__prompt_style__', hited['add_prompt_style'])
    else:
        pos_prompt = pos_prompt_with_cloth2.replace('__prompt_cloth__', prompt_cloth)
    return pos_prompt


def main_predict(reqid='iShot_2024-03-20_18.53.51.png', user_prompt_cloth='', user_prompt_style=''):
    timestamp = TimeMarker('main_predict')

    styles = []
    for base_model in base_models:
        style_in_base = []
        folder_path = f"{PROJECT_DIR}/styles/{base_model['name']}"
        files = os.listdir(folder_path)
        files.sort()
        for file in files:
            file_path = os.path.join(folder_path, file)
            with open(file_path, "r") as f:
                data = json.load(f)
                style_in_base.append(data['name'])
                styles.append(data)
        base_model['style_list'] = style_in_base

    use_main_model = True
    use_face_swap = True
    use_post_process = True
    use_stylization = False
    use_depth_control = False
    use_pose_model = False
    pose_image = 'poses/man/pose1.png'
    processed_dir = f'./data/cache_req/{reqid}/output_processed'
    num_generate = 5
    multiplier_style = 0.25
    multiplier_human = 0.85
    train_output_dir = f'./data/cache_req/{reqid}/output_train'
    output_dir = f'./data/cache_req/{reqid}/output_generated'
    base_model = base_models[0]
    style = styles[0]
    model_id = style['model_id']

    if model_id == None:
        style_model_path = None
        pos_prompt = generate_pos_prompt(
            style['name'],
            style['add_prompt_style'],
            styles,
            user_prompt_cloth=user_prompt_cloth,
            user_prompt_style=user_prompt_style
        )
    else:
        if os.path.exists(model_id):
            model_dir = model_id
        else:
            model_dir = snapshot_download_dk(model_id, revision=style['revision'])
        style_model_path = os.path.join(model_dir, style['bin_file'])
        pos_prompt = generate_pos_prompt(
            style['name'],
            style['add_prompt_style'],
            styles,
            user_prompt_cloth=user_prompt_cloth,
            user_prompt_style=user_prompt_style
        )  # style has its own prompt

    if not use_pose_model:
        pose_model_path = None
        use_depth_control = False
        pose_image = None
    else:
        model_dir = snapshot_download_dk('damo/face_chain_control_model', revision='v1.0.1')
        pose_model_path = os.path.join(model_dir, 'model_controlnet/control_v11p_sd15_openpose')

    all_params = """pose_model_path, pose_image, use_depth_control, pos_prompt, neg_prompt, style_model_path,
                               multiplier_style, multiplier_human, use_main_model,
                               use_face_swap, use_post_process,
                               use_stylization"""
    all_params = [x.strip() for x in all_params.split(',')]
    PF.print_list(all_params, 'all_params')
    all_params = dict(zip(all_params, [str(x) for x in [pose_model_path, pose_image, use_depth_control, pos_prompt, neg_prompt, style_model_path,
                                                        multiplier_style, multiplier_human, use_main_model,
                                                        use_face_swap, use_post_process,
                                                        use_stylization]]))

    PF.print_dict(all_params, title='all_params')

    gen_portrait = GenPortrait(pose_model_path, pose_image, use_depth_control, pos_prompt, neg_prompt, style_model_path,
                               multiplier_style, multiplier_human, use_main_model,
                               use_face_swap, use_post_process,
                               use_stylization)

    outputs = gen_portrait(processed_dir, num_generate, base_model['model_id'],
                           train_output_dir, base_model['sub_path'], base_model['revision'])

    os.makedirs(output_dir, exist_ok=True)
    runtime_flag = DateTime.datetime()

    for i, out_tmp in enumerate(outputs):
        cv2.imwrite(os.path.join(output_dir, f'{runtime_flag}_{i}.png'), out_tmp)

    timestamp.end()


if __name__ == '__main__':
    main_predict()
