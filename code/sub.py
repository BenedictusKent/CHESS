import subprocess
import time

last_index = 0
while True:
    try:
        f = open("/home/robotics/workspace2/team6_ws/src/send_script/send_script/command.txt", "r")
        latest_f_read = f.read()
        latest_command = latest_f_read.split("\n")[-2]
        current_index = int(latest_command.split(":")[0])
        commands = latest_command.split(":")[1].split("->")

        from_position = int(commands[0])
        to_position = int(commands[1])
        action = commands[2]

        if current_index != last_index:
            last_index = current_index

            # must run source install/setup.bash first
            p = subprocess.run(["ros2", "run" ,"send_script" ,"send_script"])
        f.close()

    except:
        print("Fail to read file")

    time.sleep(1)

