import cv2
import json
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from pynput import mouse, keyboard
from pynput.keyboard import Key, Controller

class keypoint:
    def __init__ (self, name, hand, index, data_array):
    
        global width
        global height

        self.coordinate = (int(data_array[hand][index].x * width), int(data_array[hand][index].y * height))

        self.x = data_array[hand][index].x * width
        self.y = data_array[hand][index].y * height
        self.depth = data_array[hand][index].z * -100


        self.normcoordinate = (data_array[hand][index].x, data_array[hand][index].y)
        self.normx = data_array[hand][index].x
        self.normy = data_array[hand][index].y 
        

    def drawCircle (self, radius, colour):
        if self.coordinate is not None:
            cv2.circle(frame, self.coordinate, radius = radius, color = colour, thickness =- 1)


class keyboard_key:
    def __init__(self, displaytext, output, coordinates, width = 80, height = 80):
        self.displaytext = displaytext
        self.output = output

        self.coord1 = coordinates
        self.coord2 = (coordinates[0] + width, coordinates[1] + height)

        self.x1 = coordinates[0]
        self.y1 = coordinates[1]

        self.x2 = coordinates[0] + width
        self.y2 = coordinates[1] + height

        self.width = width
        self.height = height

        self.pressed = False

    def draw(self, colour = (255, 0, 0), font = cv2.FONT_HERSHEY_SIMPLEX, size = 1.0, box_thickness = 3, text_thickness = 2):

        cv2.rectangle(frame, self.coord1, self.coord2, colour, box_thickness)

        (text_width, text_height), useless = cv2.getTextSize(self.displaytext, font, size, text_thickness)

        boxcentre = (int(self.x1 + self.width / 2), int(self.y1 + self.height / 2))
        textorigin = (int(boxcentre[0] - text_width / 2), int(boxcentre[1] + text_height / 2))

        cv2.putText(frame, self.displaytext, (textorigin), font, size, colour, thickness = text_thickness)
    

    def detect(self, detecting = "indexTip", thresh = 8):

        detected = False

        for key in keypointDict:
            if key.startswith(detecting):

                if (keypointDict[key].coordinate[0] > self.x1 
                    and keypointDict[key].coordinate[0] < self.x2 #X coordinate checks

                    and keypointDict[key].coordinate[1] > self.y1 
                    and keypointDict[key].coordinate[1] < self.y2 #Y coordinate checks

                    and keypointDict[key].depth > thresh): #Depth check  
                        
                
                        detected = True
                        break
                
        if detected and not self.pressed:

            keyboard.press(self.output)
            self.pressed = True

        if not detected and self.pressed:

            keyboard.release(self.output)
            self.pressed = False
                

def json_to_keys(string):
    if string.startswith("**"):
        print(string)
        spckey = string.lstrip("*")

        try:

            spckey = Key[spckey]
            return(spckey)
        
        except KeyError:
            for a in range (10):
                print(f"{spckey} is an invalid key")

    else:
        return string


keyboard = Controller()

task_api_path = python.BaseOptions(model_asset_path = "mediapipe_task_api.task")

detector_config = vision.HandLandmarkerOptions(
    base_options = task_api_path,
    running_mode = vision.RunningMode.IMAGE,
    num_hands = 2
)

keypoint_model = vision.HandLandmarker.create_from_options(detector_config)

camera = cv2.VideoCapture(0)

camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

width = camera.get(cv2.CAP_PROP_FRAME_WIDTH)
height = camera.get(cv2.CAP_PROP_FRAME_HEIGHT)

print((width, height))

displaydepth = True


with open("standardquertykeyboard.json", "r") as file:
    keysdata = json.load(file)


print(keysdata)

pointNameList = ["wrist", "thumb1", "thumb2", "thumb3", "thumbTip", 
                 "index1", "index2", "index3", "indexTip", 
                 "middle1", "middle2", "middle3", "middleTip", 
                 "ring1", "ring2", "ring3", "ringTip", 
                 "little1", "little2", "little3", "littleTip"]

keyDict = {}

keyInfo = []

key_id = 0

for keys in keysdata:

    text = keys["displaytext"]
    char = json_to_keys(keys["outputchar"])
    coords = tuple(keys["coordinates"]) 


    if "width" in keys and "height" in keys:
        keyInfo.append((text, char, coords, keys["width"], keys["height"]))
        keyDict[f"{key_id}_{text}"] = keyboard_key(text, char, coords, width = keys["width"], height = keys["height"])

    elif "width" in keys:
        keyInfo.append((text, char, coords, keys["width"]))
        keyDict[f"{key_id}_{text}"] = keyboard_key(text, char, coords, width = keys["width"])

    elif "height" in keys:
        keyInfo.append((text, char, coords, keys["height"]))
        keyDict[f"{key_id}_{text}"] = keyboard_key(text, char, coords, height = keys["height"])
    else:
        keyInfo.append((text, char, coords))
        keyDict[f"{key_id}_{text}"] = keyboard_key(text, char, coords)

    print(key_id)
    key_id += 1

del key_id


print(keyInfo)
print(keyDict)


while True:

    useless, frame = camera.read()

    frame = cv2.flip(frame, 1)


    frame_for_inference = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_for_inference = mp.Image(image_format = mp.ImageFormat.SRGB, data = frame_for_inference)

    detector_results = keypoint_model.detect(frame_for_inference).hand_landmarks


    data_array = []


    keypointDict = {}
    keypointList = []
    
    for key in keyDict:
        keyDict[key].draw()

    if detector_results is not None:


        for hand in detector_results:
            current_hand_data = []

            for data in hand:
                current_hand_data.append(data)

            data_array.append(current_hand_data)


        try:
            for a in range(len(detector_results)):
                for i in range(21):
                    keypointDict[f"{pointNameList[i]}_{a + 1}"] = keypoint(pointNameList[i], a, i, data_array)
                    keypointList.append(keypointDict[f"{pointNameList[i]}_{a + 1}"])

            for key in keypointDict:
                if key.startswith("wrist"):
                    keypointDict[key].drawCircle(15, (255, 255, 255))

                elif key.startswith("thumb"):
                    keypointDict[key].drawCircle(5, (255, 0, 0))

                elif key.startswith("index"):
                    keypointDict[key].drawCircle(5, (255, 255, 0))

                elif key.startswith("middle"):
                    keypointDict[key].drawCircle(5, (0, 255, 0))

                elif key.startswith("ring"):
                    keypointDict[key].drawCircle(5, (0, 255, 255))

                elif key.startswith("little"):
                    keypointDict[key].drawCircle(5, (0, 0, 255))


            if displaydepth:
                for key in keypointDict:
                    if keypointDict[key].depth < 8:
                        cv2.putText(frame, str(round(keypointDict[key].depth, 3)), keypointDict[key].coordinate, cv2.FONT_HERSHEY_COMPLEX, 1.2, (0, 255, 0))
                    if keypointDict[key].depth > 8:
                        cv2.putText(frame, str(round(keypointDict[key].depth, 3)), keypointDict[key].coordinate, cv2.FONT_HERSHEY_COMPLEX, 1.2, (0, 0, 255))


            for key in keyDict:
                keyDict[key].detect()

        except (IndexError, KeyError, AttributeError) as e:
            print(f"No valid Result {e}")


    cv2.imshow("Screen", frame)

    if cv2.waitKey(1) == ord("e"):

        cv2.destroyAllWindows()
        break


