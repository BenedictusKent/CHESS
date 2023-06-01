from utils import *
from google.cloud import speech
import sounddevice as sd
from scipy.io.wavfile import write
import soundfile as sf
from gtts import gTTS
from playsound import playsound
import requests

red, black = 0, 1
go , eat = 0, 1

board, dead_pieces, alive_pieces = initialize()

def perform(source, target, action, turn):
    source_piece = board[source[0]][source[1]]
    text = ""
    
    #invalid actions
    if source_piece == None:
        text = "Invalid action, there is no piece to move!"
        print(text)
        return -1, text
    if turn != source_piece.color:
        text = "Invalid action, you can only move your pieces!"
        print(text)
        return -2, text
    if not source_piece.is_valid(target,action):
        text = "Invalid action, your piece can't reach the target!"
        print(text)
        return -3, text
    if action == eat:
        if board[target[0]][target[1]] == None:
            text = "Invalid action, there is no piece to eat!"
            print(text)
            return -4, text
        if board[target[0]][target[1]].color == source_piece.color:
            text = "Invalid action, you can't eat your ally!"
            print(text)
            return -5, text
    else:
        if not is_empty(board,target):
            text = "Invalid action, the target position is not empty!"
            print(text)
            return -6, text
    
    #need to check whether the action will cause yourself be checkmated
    if is_checkmate(alive_pieces,turn,source_piece,target):
        text = "Invalid action, your move will make you checkmate!"
        print(text)
        return -7, text
    
    #valid action
    if action == eat:
        alive_pieces[board[target[0]][target[1]].color].remove(board[target[0]][target[1]])
        dead_pieces[board[target[0]][target[1]].color].append(board[target[0]][target[1]])    
    board[source[0]][source[1]] = None
    board[target[0]][target[1]] = source_piece
    source_piece.row, source_piece.col = target[0],target[1]
    
    #arm_perform(source,target,action) # robot arm perform the task
    if is_checkmate(alive_pieces,turn):
        text = "Check"
        print("****************************************************check!****************************************************")
        if is_gameover(alive_pieces,turn):
            text = f"Game over, {'Red' if not turn else 'Black'} win!"
            print(text)
    return 100, text

def feedback(text: str):
    temp = text
    obj = gTTS(text=temp, lang='en', slow=False)
    obj.save("sound.wav")
    playsound("sound.wav")
    print(temp)

def element_exist(sentence: list, array: list):
    result = []
    for i in range(len(sentence)):
        if sentence[i] in array:
            index = array.index(sentence[i])
            result.append(array[index])
    return result

def listening():
    print("Listening...")
    # Listen audio
    freq = 44100
    duration = 6
    recording = sd.rec(int(duration * freq), samplerate=freq, channels=1)
    sd.wait()
    write("recording.wav", freq, recording)

    print("Writing...")
    # Change  PCM signed 32-bit -> PCM signed 16-bit
    data, samplerate = sf.read("recording.wav")
    sf.write("recording.wav", data, samplerate, subtype='PCM_16')

def recognize_speech():
    print("Connecting...")
    # Connect to API key
    client = speech.SpeechClient.from_service_account_file("speech-to-text-key.json")

    print("Reading...")
    # Recorded audio
    filename = "recording.wav"
    with open(filename, 'rb') as f:
        data = f.read()

    # Speech recognition
    audio = speech.RecognitionAudio(content=data)
    config = speech.RecognitionConfig(
        # encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=44100,
        language_code="en-US",
        enable_automatic_punctuation=False,
        speech_contexts=[{
            "phrases": ['quit', 'command', 'go', 'eat', 'alpha', 'bravo', 'charlie', 'delta', 'echo', 'foxtrot', 'golf', 'hotel', 'india', 'juliett', '1', '2', '3', '4', '5', '6', '7', '8', '9'  ],
            "boost": 10.0
        }],
    )
    response = client.recognize(config=config, audio=audio)

    text = ""

    # Result
    for result in response.results:
        text = result.alternatives[0].transcript
        text = text.lower()
        print("Transcript:", text)

    return text

def post_api(from_pos, to_pos, action):
    postObj = {'action': "{0}->{1}->{2}".format(from_pos, to_pos, action)}
    x = requests.post("http://159.65.131.166:5000/action", json = postObj)
    return


if __name__ == "__main__":
    showboard(board, dead_pieces)
    turn = black
    repeat = 0
    while True:
        if repeat == 999:
            break

        turn_str = "Black"  if turn == black else "Red"
        listening()
        text = recognize_speech()

        # Check result
        if "command" in text:
            # All movements are assumed to be after the word 'command'
            repeat = 0
            text = text.split()
            index = text.index("command")
            text = text[index:]
            while repeat < 3:
                repeat += 1
                feedback("What do you want to do?")
                listening()
                text = recognize_speech()
                text = text.split()

                # Check if movement is a complete set
                if len(text) == 5:
                    moves = ['go', 'eat']
                    ypos = ['alpha', 'bravo', 'charlie', 'delta', 'echo', 'foxtrot', 'golf', 'hotel', 'india', 'juliett']
                    xpos = ['1', '2', '3', '4', '5', '6', '7', '8', '9']

                    # Check if sentence has all the required data
                    temp1 = element_exist(text, moves)
                    temp2 = element_exist(text, ypos)
                    temp3 = element_exist(text, xpos)

                    if (len(temp1) == 1) and (len(temp2) ==2) and (len(temp3) == 2):
                        # Check validity
                        VALID = False
                        MOVE = temp1[0]
                        YPOS1 = 9 - ypos.index(temp2[0])
                        XPOS1 = xpos.index(temp3[0]) 
                        YPOS2 = 9 - ypos.index(temp2[1])
                        XPOS2 = xpos.index(temp3[1]) 
                        command = "{0},{1},{2},{3},{4}".format(YPOS1, XPOS1, MOVE, YPOS2, XPOS2)
                        source_row, source_col, action, target_row, target_col = eval(command)
                        source = source_row, source_col
                        target = target_row, target_col
                        number, temp = perform(source,target,action,turn)
                        temp = temp.lower()
                        if number == 100:
                            VALID = True
                        else:
                            feedback(temp)
                        # Send to API
                        if VALID:
                            showboard(board,dead_pieces)
                            turn = 1 - turn
                            from_post = "{0}{1}".format(xpos.index(temp3[0]), ypos.index(temp2[0]))
                            to_post = "{0}{1}".format(xpos.index(temp3[1]), ypos.index(temp2[1]))
                            post_api(from_post, to_post, temp1[0])
                            feedback("I got your command! I will move it now!")
                            if "check" in temp:
                                feedback("Check!")
                            elif "game over" in temp:
                                feedback(temp)
                                repeat = 999
                            repeat = 5
                    else:
                        feedback("This move is invalid. Please try again!")
                else:
                    feedback("I didn't get that, can you repeat it again?")
        elif "quit" in text:
            feedback("Bye bye!")
            break
        else:
            print("still listening")
