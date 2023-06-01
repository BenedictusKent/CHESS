import sseclient

endpoint = "http://159.65.131.166:5000/listen"

messages = sseclient.SSEClient(endpoint)

index = 0

for msg in messages:
    commands = msg
    print("Get new message")

    f = open("/home/robotics/workspace2/team6_ws/src/send_script/send_script/command.txt", "a")
    f.write("{0}: {1}\n".format(index, commands))

    index += 1
    f.close()