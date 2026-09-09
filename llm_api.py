#llm_api
from openai import OpenAI
import re

class LLM:
    def __init__(self, coco_labels):
        self.coco_labels = coco_labels  # 挑出句子中包含coco類別的詞
        #self.label = input('請輸入指令: ')
        self.label = None
        
    def set_label(self, label):
        self.label = label
        
    def run(self):
        string = self.label

        client = OpenAI(
            base_url="base_url",  # 例如 http://localhost:8000/v1
            api_key="api_key"
        )
        response = client.chat.completions.create(
            model="llama4-scout",  # 例如 gpt-3.5-turbo、llama3-8b-instruct
            messages = [
                {"role": "system", 
                 "content": (
                     "你是一個語言分析工具。請從使用者輸入的句子中，找出與以下 COCO 類別清單相符的詞，"
                     "並且嚴格回傳一個 **JSON 陣列格式的字串**（每個詞用雙引號包起來，例如：[\"apple\", \"orange\"]）。"
                     "不要加入任何解釋或其他文字。\n"
                     f"COCO 類別如下：{', '.join(self.coco_labels)}"
                     )
                 },
                {"role": "user", "content": string}
            ]
        )

        results = response.choices[0].message.content
        llm_result = re.findall(r"""['"]([^'"]+)['"]""", results)
        return llm_result


if __name__ == "__main__":
    coco_labels = [
        "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
        "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog",
        "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
        "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
        "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
        "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
        "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant",
        "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
        "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
        "teddy bear", "hair drier", "toothbrush"
    ]
    llm = LLM(coco_labels)
    label = input("請輸入指令: ")
    llm.set_label(label)
    label = llm.run()
    print(label)
