import os
import sys
sys.path.append(".")
sys.path.append("..")

MODEL = {
    'self_touch' : 'v1',
    'hand_regard' : 'v2',
}

SCENE = {
    'base' : None,
    'crib' : 'crib.xml',
    'cubes' : 'cubes.xml',
}

POSITION = {
    'base' : 'pos="0 0 .1" euler="0 -90 0"',
    'crib' : 'pos="0 0 0.3" euler="0 -90 0"',
    'cubes' : 'pos="0 0 0.0467196" quat="0.885453 -0.000184209 -0.464728 -0.000527509"',
}

CONSTRAINTS = {
    'base' : "",
    'crib' : """
        <equality>
            <weld body1="lower_body"/>
        </equality>
        """,
    'cubes' : """
        <equality>
            <weld body1="mimo_location"/>
            <joint joint1="robot:hip_bend1" polycoef="0.532 0 0 0 0"/>
            <joint joint1="robot:hip_lean1" polycoef="0 0 0 0 0"/>
            <joint joint1="robot:hip_rot1" polycoef="0 0 0 0 0"/>
            <joint joint1="robot:hip_bend2" polycoef="0.532 0 0 0 0"/>
            <joint joint1="robot:hip_lean2" polycoef="0 0 0 0 0"/>
            <joint joint1="robot:hip_rot2" polycoef="0 0 0 0 0"/>
            <joint joint1="robot:head_tilt" polycoef="0.4 0 0 0 0"/>
            <joint joint1="robot:head_tilt_side" polycoef="0 0 0 0 0"/>
            <joint joint1="robot:right_hip1" polycoef="-1.22 0 0 0 0"/>
            <joint joint1="robot:right_hip2" polycoef="-0.70 0 0 0 0"/>
            <joint joint1="robot:right_hip3" polycoef="0.538 0 0 0 0"/>
            <joint joint1="robot:right_knee" polycoef="-1.45 0 0 0 0"/>
            <joint joint1="robot:left_hip1" polycoef="-1.41 0 0 0 0"/>
            <joint joint1="robot:left_hip2" polycoef="-0.823 0 0 0 0"/>
            <joint joint1="robot:left_hip3" polycoef="0.612 0 0 0 0"/>
            <joint joint1="robot:left_knee" polycoef="-2.14 0 0 0 0"/>
        </equality>
        """,
}

def build(config=None, path_to_assets='./MIMo/mimoEnv/assets/'):

    try:
        behavior = config['behavior']
        scene = config['scene']
        actuation_model = config['actuation_model']
        for body_part in ['body','head','eyes','arms','legs','feet','hands','fingers']:
            config[f'act_{body_part}']
    except Exception as e:
        print("Missing mandatory options in config file")
        print(e)
        exit(1)
    
    # Initialize XML
    XML = """<mujoco model="MIMo">\n<worldbody>\n"""

    # Add MIMo model
    XML += f'<body name="mimo_location" {POSITION[scene]}>\n'
    XML += '<freejoint name="mimo_location"/>\n'
    XML += f'<include file="{path_to_assets}/babybench/mimo_{MODEL[behavior]}.xml"></include>\n'

    # Add BabyBench base
    XML += f'</body>\n</worldbody>\n<include file="{path_to_assets}/babybench/base.xml"></include>\n'

    # Add BabyBench meta
    XML += f'<include file="{path_to_assets}/babybench/meta_{MODEL[behavior]}.xml"></include>\n'

    # Add BabyBench scene
    if scene != 'base':
        XML += f'<include file="{path_to_assets}/babybench/{SCENE[scene]}"></include>\n'

    # Add active actuators
    XML += '<actuator>\n'
    
    if (actuation_model == 'spring_damper') or (actuation_model == 'muscle'):

        if config['act_body'] is True:
            XML += """
            <motor class="mimo"	name="act:hip_bend" 	joint="robot:hip_bend1" gear="10.58" 	forcerange="-1 .765"	user=".8574 32.93 22.97"/>
            <motor class="mimo"	name="act:hip_twist" 	joint="robot:hip_rot1" 	gear="3.63" 	forcerange="-1 1"		user="1.699 8.136 8.136"/>
            <motor class="mimo"	name="act:hip_lean" 	joint="robot:hip_lean1" gear="7.25" 	forcerange="-1 1"		user="1.278 1.264 1.264"/>
            """
        if config['act_head'] is True:
            XML += """
            <motor class="mimo"	name="act:head_swivel" 		joint="robot:head_swivel" 		gear="1.8" 	forcerange="-1 1"		user=".6665 24.87 24.87"/>
            <motor class="mimo"	name="act:head_tilt" 		joint="robot:head_tilt" 		gear="3.3" 	forcerange="-1 .55"		user=".9338 31.39 16.86"/>
            <motor class="mimo"	name="act:head_tilt_side" 	joint="robot:head_tilt_side" 	gear="1.8" 	forcerange="-1 1"		user="1.068 15.69 15.69"/>
            """
        if config['act_eyes'] is True:
            XML += """
            <motor class="mimo" name="act:left_eye_horizontal" 				joint="robot:left_eye_horizontal" 		gear=".0054" 		forcerange="-1 1"		user="5.110 .03025 .03025"/>
            <motor class="mimo" name="act:left_eye_vertical" 				joint="robot:left_eye_vertical" 		gear=".0054" 		forcerange="-1 1"		user="6.179 .02619 .02771"/>
            <motor class="mimo" name="act:left_eye_torsional" 				joint="robot:left_eye_torsional" 		gear=".0037" 		forcerange="-1 1"		user="25.03 .003686 .003686"/>
            <motor class="mimo" name="act:right_eye_horizontal" 			joint="robot:right_eye_horizontal" 		gear=".0054" 		forcerange="-1 1"		user="5.110 .03025 .03025"/>
            <motor class="mimo" name="act:right_eye_vertical" 				joint="robot:right_eye_vertical" 		gear=".0054" 		forcerange="-1 1"		user="6.179 .02619 .02771"/>
            <motor class="mimo" name="act:right_eye_torsional" 				joint="robot:right_eye_torsional" 		gear=".0037" 		forcerange="-1 1"		user="25.03 .003686 .003686"/>
            """
        if config['act_arms'] is True:
            XML += """
            <motor class="mimo"	name="act:right_shoulder_horizontal" 		joint="robot:right_shoulder_horizontal" gear="1.8" 			forcerange="-1 1"		user="1.811 18.80 15.30"/>
            <motor class="mimo"	name="act:right_shoulder_abduction" 		joint="robot:right_shoulder_ad_ab" 		gear="4" 			forcerange="-.6875 1"	user=".8793 44.17 69.23"/>
            <motor class="mimo"	name="act:right_shoulder_internal" 			joint="robot:right_shoulder_rotation" 	gear="2.5" 			forcerange="-1 .64"		user="1.498 25.09 17.10"/>
            <motor class="mimo"	name="act:right_elbow" 						joint="robot:right_elbow" 				gear="3.6" 			forcerange="-1 .83"		user="1.566 35.43 27.04"/>
            <motor class="mimo"	name="act:left_shoulder_horizontal" 		joint="robot:left_shoulder_horizontal" 	gear="1.8" 			forcerange="-1 1"		user="1.811 18.80 15.30"/>
            <motor class="mimo"	name="act:left_shoulder_abduction" 			joint="robot:left_shoulder_ad_ab" 		gear="4" 			forcerange="-.6875 1"	user=".8793 44.17 69.23"/>
            <motor class="mimo"	name="act:left_shoulder_internal" 			joint="robot:left_shoulder_rotation" 	gear="2.5" 			forcerange="-1 .64"		user="1.498 25.09 17.10"/>
            <motor class="mimo"	name="act:left_elbow" 						joint="robot:left_elbow" 				gear="3.6" 			forcerange="-1 .83"		user="1.566 35.43 27.04"/>
            """
        if config['act_hands'] is True:
            XML += """
            <motor class="mimo"	name="act:right_wrist_rotation" 			joint="robot:right_hand1" 				gear=".7"			forcerange="-1 1"		user="1.513 7.842 7.842"/>
            <motor class="mimo"	name="act:right_wrist_flexion" 				joint="robot:right_hand2" 				gear="1.24"			forcerange="-1 .57"		user="1.455 13.66 7.874"/>
            <motor class="mimo"	name="act:right_wrist_ulnar" 				joint="robot:right_hand3" 				gear=".95"			forcerange="-.87 1"		user="2.254 5.155 6.021"/>
            <motor class="mimo"	name="act:left_wrist_rotation" 				joint="robot:left_hand1"  				gear=".7"			forcerange="-1 1"		user="1.513 7.842 7.842"/>
            <motor class="mimo"	name="act:left_wrist_flexion" 				joint="robot:left_hand2"  				gear="1.24"			forcerange="-1 .57"		user="1.455 13.66 7.874"/>
            <motor class="mimo"	name="act:left_wrist_ulnar" 				joint="robot:left_hand3"  				gear=".95"			forcerange="-.87 1"		user="2.254 5.155 6.021"/>
            """
        if config['act_fingers'] is True:
            if MODEL[behavior] == 'v1':
                XML += """
                <motor class="mimo" name="act:right_fingers" 					joint="robot:right_fingers" 			gear=".69"			forcerange="-1 .33" 	user="3.019 6.854 2.551"/>
                <motor class="mimo" name="act:left_fingers" 					joint="robot:left_fingers" 				gear=".69" 			forcerange="-1 .33" 	user="3.019 6.854 2.551"/>
                """
            elif MODEL[behavior] == 'v2':
                XML += """
                <motor class="mimo"	name="act:right_ff_side" 			 		joint="robot:right_ff_side" 			gear=".05"			forcerange="-1 1"		user="6.217 .1556 .1556"/>
                <motor class="mimo"	name="act:right_ff_knuckle" 			 	joint="robot:right_ff_knuckle"			gear=".0619"		forcerange="-.345 1"	user="4.368 .1422 .4378"/>
                <motor class="mimo"	name="act:right_ff_middle" 				 	joint="robot:right_ff_middle"			gear=".02063"		forcerange="-.345 1"	user="3.493 .04684 .1339"/>
                <motor class="mimo"	name="act:right_ff_distal" 				 	joint="robot:right_ff_distal"			gear=".01548"		forcerange="-.345 1"	user="2.539 .03054 .07747"/>
                <motor class="mimo"	name="act:right_mf_side" 			 		joint="robot:right_mf_side" 			gear=".05"			forcerange="-1 1"		user="5.375 .1556 .1556"/>
                <motor class="mimo"	name="act:right_mf_knuckle" 			 	joint="robot:right_mf_knuckle"			gear=".117"			forcerange="-.345 1"	user="4.325 .2687 .8275"/>
                <motor class="mimo"	name="act:right_mf_middle" 			 		joint="robot:right_mf_middle"			gear=".039"			forcerange="-.345 1"	user="3.375 .08855 .2532"/>
                <motor class="mimo"	name="act:right_mf_distal" 				 	joint="robot:right_mf_distal"			gear=".02925"		forcerange="-.345 1"	user="2.577 .05771 .1464"/>
                <motor class="mimo"	name="act:right_rf_side"  					joint="robot:right_rf_side" 			gear=".05"			forcerange="-1 1"		user="4.984 .1556 .1556"/>
                <motor class="mimo"	name="act:right_rf_knuckle" 			 	joint="robot:right_rf_knuckle"			gear=".078"			forcerange="-.345 1"	user="2.817 .1791 .5517"/>
                <motor class="mimo"	name="act:right_rf_middle" 			 		joint="robot:right_rf_middle" 			gear=".026"			forcerange="-.345 1"	user="3.007 .05903 .1688"/>
                <motor class="mimo"	name="act:right_rf_distal" 				 	joint="robot:right_rf_distal" 			gear=".0195"		forcerange="-.345 1"	user="2.313 .03847 .09759"/>
                <motor class="mimo"	name="act:right_lf_side" 					joint="robot:right_lf_side" 			gear=".05" 			forcerange="-1 1"		user="5.282 .1556 .1556"/>
                <motor class="mimo"	name="act:right_lf_meta" 					joint="robot:right_lf_meta" 			gear=".064" 		forcerange="-.345 1"	user="6.855 .07962 .1508"/>
                <motor class="mimo"	name="act:right_lf_knuckle" 			 	joint="robot:right_lf_knuckle" 			gear=".0196"		forcerange="-.345 1"	user="2.337 .04502 .1386"/>
                <motor class="mimo"	name="act:right_lf_middle" 				 	joint="robot:right_lf_middle" 			gear=".00653"		forcerange="-.345 1"	user="2.566 .01483 .04239"/>
                <motor class="mimo"	name="act:right_lf_distal" 				 	joint="robot:right_lf_distal" 			gear=".00490"		forcerange="-.345 1"	user="2.001 .009666 .02452"/>
                <motor class="mimo"	name="act:right_thumb_side" 			 	joint="robot:right_th_swivel" 			gear=".373"			forcerange="-.345 1"	user="3.491 1.117 2.604"/>
                <motor class="mimo"	name="act:right_thumb_add" 					joint="robot:right_th_adduction" 		gear=".134" 		forcerange="-1 1"		user="5.646 .4969 .5024"/>
                <motor class="mimo"	name="act:right_thumb_pivot" 				joint="robot:right_th_pivot" 			gear=".1" 			forcerange="-1 1"		user="9.055 .1868 .1868"/>
                <motor class="mimo"	name="act:right_thumb_middle" 			 	joint="robot:right_th_middle" 			gear=".126"			forcerange="-.345 1"	user="3.877 .2433 .7065"/>
                <motor class="mimo"	name="act:right_thumb_distal" 			 	joint="robot:right_th_distal" 			gear=".0945"		forcerange="-.345 1"	user="3.392 .2053 .4959"/>
                <motor class="mimo"	name="act:left_ff_side" 		 			joint="robot:left_ff_side" 				gear=".05"			forcerange="-1 1"		user="6.217 .1556 .1556"/>
                <motor class="mimo"	name="act:left_ff_knuckle" 				 	joint="robot:left_ff_knuckle" 			gear=".0619"		forcerange="-.345 1"	user="4.368 .1422 .4378"/>
                <motor class="mimo"	name="act:left_ff_middle" 				 	joint="robot:left_ff_middle" 			gear=".02063"		forcerange="-.345 1"	user="3.493 .04684 .1339"/>
                <motor class="mimo"	name="act:left_ff_distal" 				 	joint="robot:left_ff_distal" 			gear=".01548"		forcerange="-.345 1"	user="2.539 .03054 .07747"/>
                <motor class="mimo"	name="act:left_mf_side" 		 			joint="robot:left_mf_side" 				gear=".05"			forcerange="-1 1"		user="5.375 .1556 .1556"/>
                <motor class="mimo"	name="act:left_mf_knuckle" 			 		joint="robot:left_mf_knuckle" 			gear=".117"			forcerange="-.345 1"	user="4.325 .2687 .8275"/>
                <motor class="mimo"	name="act:left_mf_middle" 			 		joint="robot:left_mf_middle" 			gear=".039"			forcerange="-.345 1"	user="3.375 .08855 .2532"/>
                <motor class="mimo"	name="act:left_mf_distal" 				 	joint="robot:left_mf_distal" 			gear=".02925"		forcerange="-.345 1"	user="2.577 .05771 .1464"/>
                <motor class="mimo"	name="act:left_rf_side" 		 			joint="robot:left_rf_side" 				gear=".05"			forcerange="-1 1"		user="4.984 .1556 .1556"/>
                <motor class="mimo"	name="act:left_rf_knuckle" 			 		joint="robot:left_rf_knuckle" 			gear=".078"			forcerange="-.345 1"	user="2.817 .1791 .5517"/>
                <motor class="mimo"	name="act:left_rf_middle" 			 		joint="robot:left_rf_middle" 			gear=".026"			forcerange="-.345 1"	user="3.007 .05903 .1688"/>
                <motor class="mimo"	name="act:left_rf_distal" 			 		joint="robot:left_rf_distal" 			gear=".0195"		forcerange="-.345 1"	user="2.313 .03847 .09759"/>
                <motor class="mimo"	name="act:left_lf_side" 		 			joint="robot:left_lf_side" 				gear=".05"			forcerange="-1 1"		user="5.282 .1556 .1556"/>
                <motor class="mimo"	name="act:left_lf_meta" 			 		joint="robot:left_lf_meta" 				gear=".064"			forcerange="-.345 1"	user="6.855 .07962 .1508"/>
                <motor class="mimo"	name="act:left_lf_knuckle" 				 	joint="robot:left_lf_knuckle" 			gear=".0196"		forcerange="-.345 1"	user="2.337 .04502 .1386"/>
                <motor class="mimo"	name="act:left_lf_middle" 				 	joint="robot:left_lf_middle" 			gear=".00653"		forcerange="-.345 1"	user="2.566 .01483 .04239"/>
                <motor class="mimo"	name="act:left_lf_distal" 				 	joint="robot:left_lf_distal" 			gear=".00490"		forcerange="-.345 1"	user="2.001 .009666 .02452"/>
                <motor class="mimo"	name="act:left_thumb_side" 			 		joint="robot:left_th_swivel" 			gear=".373"			forcerange="-.345 1"	user="3.491 1.117 2.604"/>
                <motor class="mimo"	name="act:left_thumb_add" 					joint="robot:left_th_adduction" 		gear=".134" 		forcerange="-1 1"		user="5.646 .4969 .5024"/>
                <motor class="mimo"	name="act:left_thumb_pivot" 				joint="robot:left_th_pivot" 			gear=".1" 			forcerange="-1 1"		user="9.055 .1868 .1868"/>
                <motor class="mimo"	name="act:left_thumb_middle" 			 	joint="robot:left_th_middle" 			gear=".126"			forcerange="-.345 1"	user="3.877 .2433 .7065"/>
                <motor class="mimo"	name="act:left_thumb_distal" 			 	joint="robot:left_th_distal" 			gear=".0945"		forcerange="-.345 1"	user="3.392 .2053 .4959"/>
                """
        if config['act_legs'] is True:
            XML += """
            <motor class="mimo"	name="act:right_hip_flex" 					joint="robot:right_hip1" 				gear="8" 			forcerange="-1 1"		user="1.083 71.25 92.49"/>
            <motor class="mimo"	name="act:right_hip_abduction" 				joint="robot:right_hip2" 				gear="6.24" 		forcerange="-1 1"		user="1.488 24.81 29.20"/>
            <motor class="mimo"	name="act:right_hip_rotation" 				joint="robot:right_hip3" 				gear="3.54" 		forcerange="-.75 1"		user="1.860 12.32 15.78"/>
            <motor class="mimo"	name="act:right_knee" 						joint="robot:right_knee" 				gear="10" 			forcerange="-.65 1"		user="1.450 63.16 89.27"/>
            <motor class="mimo"	name="act:left_hip_flex" 					joint="robot:left_hip1" 				gear="8" 			forcerange="-1 1"		user="1.083 71.25 92.49"/>
            <motor class="mimo"	name="act:left_hip_abduction" 				joint="robot:left_hip2" 				gear="6.24" 		forcerange="-1 1"		user="1.488 24.81 29.20"/>
            <motor class="mimo"	name="act:left_hip_rotation" 				joint="robot:left_hip3"					gear="3.54" 		forcerange="-.75 1"		user="1.860 12.32 15.78"/>
            <motor class="mimo"	name="act:left_knee" 						joint="robot:left_knee" 				gear="10" 			forcerange="-.65 1"		user="1.450 63.16 89.27"/>
            """
        if config['act_feet'] is True:
            XML += """
            <motor class="mimo"	name="act:right_foot_flexion" 				joint="robot:right_foot1" 				gear="3.78"			forcerange="-1 .5"		user="1.430 21.34 11.87"/>
            <motor class="mimo"	name="act:right_foot_inversion" 			joint="robot:right_foot2" 				gear="1.16"			forcerange="-.91 1"		user="1.988 4.184 4.645"/>
            <motor class="mimo"	name="act:right_foot_rotation" 				joint="robot:right_foot3"	 			gear="1.2"			forcerange="-1 1"		user="2.688 3.868 3.625"/>
            <motor class="mimo"	name="act:right_toes" 						joint="robot:right_toes"	 			gear=".33"			forcerange="-1 .3" 		user="1.290 2.947 .8440"/>
            <motor class="mimo"	name="act:left_foot_flexion" 				joint="robot:left_foot1" 				gear="3.78"			forcerange="-1 .5"		user="1.430 21.34 11.87"/>
            <motor class="mimo"	name="act:left_foot_inversion" 				joint="robot:left_foot2" 				gear="1.16"			forcerange="-.91 1"		user="1.988 4.184 4.645"/>
            <motor class="mimo"	name="act:left_foot_rotation" 				joint="robot:left_foot3" 				gear="1.2"			forcerange="-1 1"		user="2.688 3.868 3.625"/>
            <motor class="mimo"	name="act:left_toes" 						joint="robot:left_toes"	 				gear=".33"			forcerange="-1 .3" 		user="1.290 2.947 .8440"/>
            """
            if MODEL[behavior] == 'v2':
                XML += """
                <motor class="mimo"	name="act:right_big_toe" 					joint="robot:right_big_toe"	 			gear=".165"			forcerange="-1 .3"		user="1.774 1.473 .4220"/>
                <motor class="mimo"	name="act:left_big_toe" 					joint="robot:left_big_toe"	 			gear=".165"			forcerange="-1 .3"		user="1.805 1.473 .4220"/>
                """
    
    elif (actuation_model == 'positional'):

        if config['act_body'] is True:
            XML += """
            <position class="mimo"	name="act:hip_bend" 	joint="robot:hip_bend1"     ctrlrange="-17 30.5"	forcelimited="false"/>
            <position class="mimo"	name="act:hip_twist" 	joint="robot:hip_rot1"      ctrlrange="-18 18"	    forcelimited="false"/>
            <position class="mimo"	name="act:hip_lean" 	joint="robot:hip_lean1"     ctrlrange="-14 14"	    forcelimited="false"/>
            """
        if config['act_head'] is True:
            XML += """
            <position class="mimo"	name="act:head_swivel" 		joint="robot:head_swivel" 		ctrlrange="-111 111"	forcelimited="false"/>
            <position class="mimo"	name="act:head_tilt" 		joint="robot:head_tilt" 		ctrlrange="-70 81"		forcelimited="false"/>
            <position class="mimo"	name="act:head_tilt_side" 	joint="robot:head_tilt_side" 	ctrlrange="-70 70"		forcelimited="false"/>
            """
        if config['act_eyes'] is True:
            XML += """
            <position class="mimo" name="act:left_eye_horizontal" 		joint="robot:left_eye_horizontal" 		ctrlrange="-45 45"		forcelimited="false"/>
            <position class="mimo" name="act:left_eye_vertical" 		joint="robot:left_eye_vertical" 		ctrlrange="-47 33"		forcelimited="false"/>
            <position class="mimo" name="act:left_eye_torsional" 		joint="robot:left_eye_torsional" 		ctrlrange="-8 8"		forcelimited="false"/>
            <position class="mimo" name="act:right_eye_horizontal" 		joint="robot:right_eye_horizontal" 		ctrlrange="-45 45"		forcelimited="false"/>
            <position class="mimo" name="act:right_eye_vertical" 		joint="robot:right_eye_vertical" 		ctrlrange="-47 33"		forcelimited="false"/>
            <position class="mimo" name="act:right_eye_torsional" 		joint="robot:right_eye_torsional" 		ctrlrange="-8 8"		forcelimited="false"/>
            """
        if config['act_arms'] is True:
            XML += """
            <position class="mimo"	name="act:right_shoulder_horizontal" 	joint="robot:right_shoulder_horizontal" ctrlrange="-28 118"		forcelimited="false"/>
            <position class="mimo"	name="act:right_shoulder_abduction" 	joint="robot:right_shoulder_ad_ab" 		ctrlrange="-84 183"		forcelimited="false"/>
            <position class="mimo"	name="act:right_shoulder_internal" 		joint="robot:right_shoulder_rotation" 	ctrlrange="-99 67"		forcelimited="false"/>
            <position class="mimo"	name="act:right_elbow" 					joint="robot:right_elbow" 				ctrlrange="-146 5"		forcelimited="false"/>
            <position class="mimo"	name="act:left_shoulder_horizontal" 	joint="robot:left_shoulder_horizontal" 	ctrlrange="-28 118"		forcelimited="false"/>
            <position class="mimo"	name="act:left_shoulder_abduction" 		joint="robot:left_shoulder_ad_ab" 		ctrlrange="-84 183"		forcelimited="false"/>
            <position class="mimo"	name="act:left_shoulder_internal" 		joint="robot:left_shoulder_rotation" 	ctrlrange="-99 67"		forcelimited="false"/>
            <position class="mimo"	name="act:left_elbow" 					joint="robot:left_elbow" 				ctrlrange="-146 5"		forcelimited="false"/>
            """
        if config['act_hands'] is True:
            XML += """
            <position class="mimo"	name="act:right_wrist_rotation" 		joint="robot:right_hand1" 			ctrlrange="-90 90"		forcelimited="false"/>
            <position class="mimo"	name="act:right_wrist_flexion" 			joint="robot:right_hand2" 			ctrlrange="-92 86"		forcelimited="false"/>
            <position class="mimo"	name="act:right_wrist_ulnar" 			joint="robot:right_hand3" 			ctrlrange="-53 48"		forcelimited="false"/>
            <position class="mimo"	name="act:left_wrist_rotation" 			joint="robot:left_hand1"  			ctrlrange="-90 90"		forcelimited="false"/>
            <position class="mimo"	name="act:left_wrist_flexion" 			joint="robot:left_hand2"  			ctrlrange="-92 86"		forcelimited="false"/>
            <position class="mimo"	name="act:left_wrist_ulnar" 			joint="robot:left_hand3"  			ctrlrange="-53 48"		forcelimited="false"/>
            """
        if config['act_fingers'] is True:
            if MODEL[behavior] == 'v1':
                XML += """
                <position class="mimo" name="act:right_fingers" 				joint="robot:right_fingers" 			ctrlrange="-160 8"		forcelimited="false"/>
                <position class="mimo" name="act:left_fingers" 					joint="robot:left_fingers" 				ctrlrange="-160 8"		forcelimited="false"/>
                """
            elif MODEL[behavior] == 'v2':
                XML += """
                <position class="mimo"	name="act:right_ff_side" 			 		joint="robot:right_ff_side" 			ctrlrange="-25 25"		forcelimited="false"/>
                <position class="mimo"	name="act:right_ff_knuckle" 			 	joint="robot:right_ff_knuckle"			ctrlrange="-20 90"		forcelimited="false"/>
                <position class="mimo"	name="act:right_ff_middle" 				 	joint="robot:right_ff_middle"			ctrlrange="-5 100"		forcelimited="false"/>
                <position class="mimo"	name="act:right_ff_distal" 				 	joint="robot:right_ff_distal"			ctrlrange="-5 80"		forcelimited="false"/>
                <position class="mimo"	name="act:right_mf_side" 			 		joint="robot:right_mf_side" 			ctrlrange="-25 25"		forcelimited="false"/>
                <position class="mimo"	name="act:right_mf_knuckle" 			 	joint="robot:right_mf_knuckle"			ctrlrange="-20 90"		forcelimited="false"/>
                <position class="mimo"	name="act:right_mf_middle" 			 		joint="robot:right_mf_middle"			ctrlrange="-5 100"		forcelimited="false"/>
                <position class="mimo"	name="act:right_mf_distal" 				 	joint="robot:right_mf_distal"			ctrlrange="-5 80"		forcelimited="false"/>
                <position class="mimo"	name="act:right_rf_side"  					joint="robot:right_rf_side" 			ctrlrange="-25 25"		forcelimited="false"/>
                <position class="mimo"	name="act:right_rf_knuckle" 			 	joint="robot:right_rf_knuckle"			ctrlrange="-20 90"		forcelimited="false"/>
                <position class="mimo"	name="act:right_rf_middle" 			 		joint="robot:right_rf_middle" 			ctrlrange="-5 100"		forcelimited="false"/>
                <position class="mimo"	name="act:right_rf_distal" 				 	joint="robot:right_rf_distal" 			ctrlrange="-5 80"		forcelimited="false"/>
                <position class="mimo"	name="act:right_lf_side" 					joint="robot:right_lf_side" 			ctrlrange="-25 25"		forcelimited="false"/>
                <position class="mimo"	name="act:right_lf_meta" 					joint="robot:right_lf_meta" 			ctrlrange="0 40"		forcelimited="false"/>
                <position class="mimo"	name="act:right_lf_knuckle" 			 	joint="robot:right_lf_knuckle" 			ctrlrange="-20 90"		forcelimited="false"/>
                <position class="mimo"	name="act:right_lf_middle" 				 	joint="robot:right_lf_middle" 			ctrlrange="-5 100"		forcelimited="false"/>
                <position class="mimo"	name="act:right_lf_distal" 				 	joint="robot:right_lf_distal" 			ctrlrange="-5 80"		forcelimited="false"/>
                <position class="mimo"	name="act:right_thumb_side" 			 	joint="robot:right_th_swivel" 			ctrlrange="10 110"		forcelimited="false"/>
                <position class="mimo"	name="act:right_thumb_add" 					joint="robot:right_th_adduction" 		ctrlrange="-60 0"		forcelimited="false"/>
                <position class="mimo"	name="act:right_thumb_pivot" 				joint="robot:right_th_pivot" 			ctrlrange="-10 10"		forcelimited="false"/>
                <position class="mimo"	name="act:right_thumb_middle" 			 	joint="robot:right_th_middle" 			ctrlrange="0 90"		forcelimited="false"/>
                <position class="mimo"	name="act:right_thumb_distal" 			 	joint="robot:right_th_distal" 			ctrlrange="0 90"		forcelimited="false"/>
                <position class="mimo"	name="act:left_ff_side" 		 			joint="robot:left_ff_side" 				ctrlrange="-25 25"      forcelimited="false"/>
                <position class="mimo"	name="act:left_ff_knuckle" 				 	joint="robot:left_ff_knuckle" 			ctrlrange="-20 90"      forcelimited="false"/>
                <position class="mimo"	name="act:left_ff_middle" 				 	joint="robot:left_ff_middle" 			ctrlrange="-5 100"      forcelimited="false"/>
                <position class="mimo"	name="act:left_ff_distal" 				 	joint="robot:left_ff_distal" 			ctrlrange="-5 80"	    forcelimited="false"/>
                <position class="mimo"	name="act:left_mf_side" 		 			joint="robot:left_mf_side" 				ctrlrange="-25 25"      forcelimited="false"/>
                <position class="mimo"	name="act:left_mf_knuckle" 			 		joint="robot:left_mf_knuckle" 			ctrlrange="-20 90"      forcelimited="false"/>
                <position class="mimo"	name="act:left_mf_middle" 			 		joint="robot:left_mf_middle" 			ctrlrange="-5 100"      forcelimited="false"/>
                <position class="mimo"	name="act:left_mf_distal" 				 	joint="robot:left_mf_distal" 			ctrlrange="-5 80"	    forcelimited="false"/>
                <position class="mimo"	name="act:left_rf_side" 		 			joint="robot:left_rf_side" 				ctrlrange="-25 25"      forcelimited="false"/>
                <position class="mimo"	name="act:left_rf_knuckle" 			 		joint="robot:left_rf_knuckle" 			ctrlrange="-20 90"      forcelimited="false"/>
                <position class="mimo"	name="act:left_rf_middle" 			 		joint="robot:left_rf_middle" 			ctrlrange="-5 100"      forcelimited="false"/>
                <position class="mimo"	name="act:left_rf_distal" 			 		joint="robot:left_rf_distal" 			ctrlrange="-5 80"	    forcelimited="false"/>
                <position class="mimo"	name="act:left_lf_side" 		 			joint="robot:left_lf_side" 				ctrlrange="-25 25"      forcelimited="false"/>
                <position class="mimo"	name="act:left_lf_meta" 			 		joint="robot:left_lf_meta" 				ctrlrange="0 40"		forcelimited="false"/>
                <position class="mimo"	name="act:left_lf_knuckle" 				 	joint="robot:left_lf_knuckle" 			ctrlrange="-20 90"      forcelimited="false"/>
                <position class="mimo"	name="act:left_lf_middle" 				 	joint="robot:left_lf_middle" 			ctrlrange="-5 100"      forcelimited="false"/>
                <position class="mimo"	name="act:left_lf_distal" 				 	joint="robot:left_lf_distal" 			ctrlrange="-5 80"	    forcelimited="false"/>
                <position class="mimo"	name="act:left_thumb_side" 			 		joint="robot:left_th_swivel" 			ctrlrange="10 110"      forcelimited="false"/>
                <position class="mimo"	name="act:left_thumb_add" 					joint="robot:left_th_adduction" 		ctrlrange="-60 0"	    forcelimited="false"/>
                <position class="mimo"	name="act:left_thumb_pivot" 				joint="robot:left_th_pivot" 			ctrlrange="-10 10"      forcelimited="false"/>
                <position class="mimo"	name="act:left_thumb_middle" 			 	joint="robot:left_th_middle" 			ctrlrange="0 90"		forcelimited="false"/>
                <position class="mimo"	name="act:left_thumb_distal" 			 	joint="robot:left_th_distal" 			ctrlrange="0 90"		forcelimited="false"/>
                """
        if config['act_legs'] is True:
            XML += """
            <position class="mimo"	name="act:right_hip_flex" 					joint="robot:right_hip1" 				ctrlrange="-133 20"		forcelimited="false"/>
            <position class="mimo"	name="act:right_hip_abduction" 				joint="robot:right_hip2" 				ctrlrange="-51 17"		forcelimited="false"/>
            <position class="mimo"	name="act:right_hip_rotation" 				joint="robot:right_hip3" 				ctrlrange="-32 41"		forcelimited="false"/>
            <position class="mimo"	name="act:right_knee" 						joint="robot:right_knee" 				ctrlrange="-145 4"		forcelimited="false"/>
            <position class="mimo"	name="act:left_hip_flex" 					joint="robot:left_hip1" 				ctrlrange="-133 20"		forcelimited="false"/>
            <position class="mimo"	name="act:left_hip_abduction" 				joint="robot:left_hip2" 				ctrlrange="-51 17"		forcelimited="false"/>
            <position class="mimo"	name="act:left_hip_rotation" 				joint="robot:left_hip3"					ctrlrange="-32 41"		forcelimited="false"/>
            <position class="mimo"	name="act:left_knee" 						joint="robot:left_knee" 				ctrlrange="-145 4"		forcelimited="false"/>
            """
        if config['act_feet'] is True:
            XML += """
            <position class="mimo"	name="act:right_foot_flexion" 				joint="robot:right_foot1" 				ctrlrange="-63 32"		forcelimited="false"/>
            <position class="mimo"	name="act:right_foot_inversion" 			joint="robot:right_foot2" 				ctrlrange="-33 31"		forcelimited="false"/>
            <position class="mimo"	name="act:right_foot_rotation" 				joint="robot:right_foot3"	 			ctrlrange="-20 30"		forcelimited="false"/>
            <position class="mimo"	name="act:right_toes" 						joint="robot:right_toes"	 			ctrlrange="-60 80"		forcelimited="false"/>
            <position class="mimo"	name="act:left_foot_flexion" 				joint="robot:left_foot1" 				ctrlrange="-63 32"		forcelimited="false"/>
            <position class="mimo"	name="act:left_foot_inversion" 				joint="robot:left_foot2" 				ctrlrange="-33 31"		forcelimited="false"/>
            <position class="mimo"	name="act:left_foot_rotation" 				joint="robot:left_foot3" 				ctrlrange="-20 30"		forcelimited="false"/>
            <position class="mimo"	name="act:left_toes" 						joint="robot:left_toes"	 				ctrlrange="-60 80"		forcelimited="false"/>
            """
            if MODEL[behavior] == 'v2':
                XML += """
                <position class="mimo"	name="act:right_big_toe" 				joint="robot:right_big_toe"	 			ctrlrange="-60 80"		forcelimited="false"/>
                <position class="mimo"	name="act:left_big_toe" 				joint="robot:left_big_toe"	 			ctrlrange="-60 80"		forcelimited="false"/>
                """

    XML += '</actuator>\n'

    # Add constraints
    XML += CONSTRAINTS[scene]
    
    XML += '</mujoco>'
    return XML


if __name__ == '__main__':

    import yaml
    with open('config.yml') as f:
        config = yaml.safe_load(f)
    xml = build(config)
    print(xml)