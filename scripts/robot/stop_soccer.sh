#!/bin/bash
# ON THE ROBOT: stop the soccer programs (display, launch, every driver node it started)
# ([x] in the patterns: the grep never matches its own command line)
ps -eo pid,cmd | grep -E "[p]inky_display.py|[s]tart_display.sh" | awk '{print $1}' | xargs -r kill
ps -eo pid,cmd | grep -E "[r]obot.launch.py|[s]tart_robot.sh" | awk '{print $1}' | xargs -r kill -INT
sleep 4
ps -eo pid,cmd | grep -E "[s]llidar_node|lib/pinky_bringup/[b]ringup|[b]attery_publisher|[c]amera_publisher|[r]obot_state_publisher|[j]oint_state_publisher|[w]heel_joint_state_relay|[c]ommand_watchdog|[t]ime_sync_status|[s]tatic_transform_publisher|[b]no055|[p]inky_sensor_adc|[b]attery_state_relay|[e]nsure_led|[l]atency_echo|[l]ed_server|[r]obot.launch.py" | awk '{print $1}' | xargs -r kill -9
sleep 1
echo "left: $(ps -eo cmd | grep -cE '[s]llidar_node|[c]amera_publisher|[r]obot.launch.py|[p]inky_display.py')"
