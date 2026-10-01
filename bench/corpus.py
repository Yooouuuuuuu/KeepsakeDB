"""Benchmark corpus: the 556 endurance facts (zh-TW companion memories) with labelled
paraphrase probes, plus a generator of synthetic companion-style distractors.

Every engine receives exactly the same documents (facts + corrections as separate
memories, the way a pipeline would store them) and the same probes.
"""
from __future__ import annotations

import random
from typing import List, Tuple

FACTS = [
    "跟你說，我叫林承宇，叫我承宇就好",                       # 0
    "我的生日是七月十四號，記得幫我慶生喔",                   # 1
    "我最喜歡吃麻辣鍋，越辣越好",                             # 2
    "我有個妹妹叫昀珊，她在高雄讀大學，念設計系",             # 3
    "我在台積電上班的事你知道吧，我是在 F12 廠區",            # 4
    "我平常都騎 Gogoro 上班，白色的",                         # 5
    "我最怕蟑螂，看到會尖叫的那種",                           # 6
    "我咖啡只喝拿鐵，而且一定要燕麥奶",                       # 7
    "我大學是讀交大電機的",                                   # 8
    "假日我喜歡去爬山，最常去十八尖山",                       # 9
    "我房間有一把電吉他，Fender 的，但很久沒彈了",            # 10
    "跟你說個秘密，我其實會做菜，拿手菜是三杯雞",             # 11
    "我媽媽住在台中，她開了一間花店",                         # 12
    "我最喜歡的樂團是草東沒有派對",                           # 13
    "我有輕微乳糖不耐，所以牛奶不太能喝",                     # 14
    "我的手機是 iPhone，但電腦堅持用 Windows",                # 15
    "十一月我要跑新竹馬拉松，報了半馬",                       # 16
    "我養 Mochi 三年了，是從收容所領養的",                    # 17
    "我最喜歡的顏色是深藍色",                                 # 18
    "我睡前習慣看 YouTube 的料理影片",                        # 19
    "我的英文名字是 Marcus，以前外商實習取的",                # 20
    "我不吃香菜，拜託記住這件事",                             # 21
    "我妹養了一隻柴犬叫豆豆",                                 # 22
    "我固定每週三晚上跟大學同學打線上遊戲，玩 LOL",           # 23
    "我近視滿深的，大概六百度，平常戴隱形眼鏡",               # 24
    "公司附近那間「阿婆麵店」是我的愛店，都點乾麵加蛋",       # 25
    "我今年目標是存到五十萬",                                 # 26
    "我有收集公仔的習慣，主要是吉卜力的",                     # 27
    "我腰不太好，久坐會痠，所以買了升降桌",                   # 28
    "我最討厭下雨天，因為騎車很麻煩",                         # 29
    # corrections (stored as their own memories)
    "欸更正一下，Mochi 的生日我記錯了，是三月五號才對，不是三月三號",   # 30
    "跟你說，我換組了！從製程調到韌體部門，以後做 firmware 了",         # 31
    "日本行程改了喔，東京那趟從一週改成十天，因為多請了假",             # 32
    "我妹轉學了，現在在台南讀，不在高雄了",                             # 33
    "馬拉松我升級了，從半馬改報全馬，瘋了我",                           # 34
]

# (probe, acceptable memory indexes) — a hit = any acceptable index in the top-k
PROBES: List[Tuple[str, List[int]]] = [
    ("我叫什麼名字？", [0, 20]),
    ("我的生日是哪天？", [1]),
    ("我最愛吃的食物是什麼？", [2]),
    ("我妹妹現在在哪個城市讀書？", [33]),
    ("我在公司哪個單位、做什麼工作？", [31, 4]),
    ("我都怎麼通勤上班？", [5]),
    ("我最怕什麼東西？", [6]),
    ("我喝咖啡有什麼講究？", [7]),
    ("我大學讀哪裡什麼系？", [8]),
    ("Mochi 的生日是哪天？", [30]),
    ("我的貓是怎麼來的？養多久了？", [17]),
    ("我十一月有什麼計畫？", [34, 16]),
    ("我去日本要去多久？", [32]),
    ("我有什麼不吃的東西？", [21]),
    ("我妹妹養了什麼寵物？", [22]),
    ("我每週三晚上固定做什麼？", [23]),
    ("我最喜歡的樂團是？", [13]),
    ("我媽媽住哪裡、做什麼的？", [12]),
    ("我收集什麼東西？", [27]),
    ("我現在住在哪個城市？", [16, 9]),
    ("我騎什麼車？什麼顏色？", [5]),
    ("我有樂器嗎？", [10]),
    ("我會煮什麼菜？", [11]),
    ("我喝牛奶會怎樣？", [14]),
    ("我用什麼手機和電腦？", [15]),
    ("我最喜歡什麼顏色？", [18]),
    ("我睡前都做什麼？", [19]),
    ("我的英文名字是什麼？", [20]),
    ("我眼睛的狀況如何？", [24]),
    ("我常去哪間麵店？都點什麼？", [25]),
    ("我今年的存錢目標是多少？", [26]),
    ("我為什麼買升降桌？", [28]),
    ("我討厭什麼天氣？", [29]),
    ("我假日喜歡做什麼？去哪裡？", [9]),
    ("我妹妹念什麼科系？", [3]),
    ("我在哪間公司上班？", [4]),
    ("我以前在哪裡實習？", [20]),
    ("我的貓叫什麼名字？", [17, 30]),
    ("我換到哪個部門了？", [31]),
    ("東京要去幾天？", [32]),
]

# ----------------------------------------------------------------- distractors
_NAMES = ["小安", "阿凱", "佳玲", "宇辰", "欣怡", "柏翰", "家豪", "雅婷", "志明", "淑芬", "冠宇", "怡君",
          "Kevin", "Amy", "Jason", "Tina", "Leo", "Nina"]
_CITIES = ["台北", "桃園", "台中", "彰化", "嘉義", "台南", "屏東", "宜蘭", "花蓮", "基隆", "東京", "大阪", "首爾", "曼谷"]
_FOODS = ["滷肉飯", "牛肉麵", "鹹酥雞", "珍珠奶茶", "壽司", "拉麵", "火鍋", "牛排", "蛋餅", "芒果冰", "咖哩", "小籠包"]
_PETS = ["柴犬", "貴賓狗", "橘貓", "黑貓", "兔子", "烏龜", "鸚鵡", "倉鼠"]
_PET_NAMES = ["布丁", "可樂", "小黑", "阿肥", "妞妞", "糖糖", "歐歐", "球球"]
_HOBBIES = ["打籃球", "游泳", "看電影", "打羽球", "畫畫", "烘焙", "拍照", "釣魚", "彈鋼琴", "跳舞", "露營", "跑步"]
_JOBS = ["工程師", "設計師", "護理師", "老師", "會計", "業務", "公務員", "廚師", "行銷", "研究助理"]
_COMPANIES = ["聯發科", "鴻海", "華碩", "中華電信", "長榮", "統一", "台達電", "日月光", "宏碁", "玉山銀行"]
_BRANDS = ["Toyota", "Honda", "Tesla", "YouBike", "捷運", "公車", "Uber", "機車"]
_COLORS = ["紅色", "綠色", "黃色", "紫色", "黑色", "白色", "橘色", "粉紅色"]
_BANDS = ["五月天", "告五人", "茄子蛋", "Coldplay", "YOASOBI", "滅火器", "宇宙人", "Mayday"]
_WEEKDAYS = ["一", "二", "四", "五", "六", "日"]
_MONTHS = ["一", "二", "三", "四", "五", "六", "八", "九", "十", "十二"]

# Distractors never contradict the labelled first-person facts (a real single-user store
# would have superseded such contradictions): they are about third parties, or about
# first-person topics the probes do not ask about. Lexical/semantic overlap stays high.
_TEMPLATES = [
    "我{rel}叫{name}，住在{city}",
    "我{rel}在{city}當{job}",
    "我{rel}超愛吃{food}，一週至少吃兩次",
    "{name}不敢吃{food}，味道受不了",
    "我{rel}養了一隻{pet}叫{petname}，{num}歲了",
    "{name}養了一隻{pet}叫{petname}",
    "我{rel}假日喜歡{hobby}，通常跟朋友一起",
    "{name}最近迷上{hobby}，每週{weekday}都去",
    "我{rel}在{company}上班，當{job}",
    "我{rel}上班都{brand}，大概{num}分鐘",
    "我{rel}最喜歡的顏色是{color}",
    "{name}最喜歡的樂團是{band}",
    "我{rel}{month}月要去{city}玩{num}天",
    "我{rel}的生日是{month}月{day}號",
    "{name}今年想存到{big}萬",
    "我{rel}每週{weekday}晚上固定{hobby}",
    "我{rel}老家在{city}，過年都會回去",
    "{name}大學讀{city}的學校，念{major}",
    "我{rel}討厭{weather}，心情會很差",
    "{name}是我大學同學，現在在{company}",
    "我{rel}最近身體不太好，在{city}的醫院看診",
    "我早餐常吃{food}配{drink}",
    "我新買了一台{gadget}，花了{big}萬",
    "我最近在學{lang}，每天練{num}分鐘",
    "我上週去{city}出差，累死了",
    "我覺得{city}的{food}最好吃",
    "我最近在追一部{genre}劇，每天看兩集",
]
_GADGETS = ["筆電", "相機", "Switch", "吸塵器", "電視", "咖啡機", "平板", "螢幕"]
_GENRES = ["韓", "日", "美", "台", "泰", "英"]
_RELS = ["爸爸", "媽媽", "哥哥", "姊姊", "弟弟", "表妹", "阿姨", "室友", "男友", "女友", "同事", "老闆"]
_MAJORS = ["資工", "企管", "外文", "機械", "化工", "法律", "心理", "護理", "建築", "財金"]
_WEATHER = ["颱風天", "大熱天", "寒流", "霧霾", "梅雨季"]
_DRINKS = ["美式", "紅茶", "豆漿", "鮮奶茶", "抹茶拿鐵"]
_LANGS = ["日文", "韓文", "法文", "西班牙文", "德文"]


def distractors(n: int, seed: int = 7) -> List[str]:
    rng = random.Random(seed)
    out: List[str] = []
    seen = set()
    while len(out) < n:
        t = rng.choice(_TEMPLATES)
        s = t.format(
            rel=rng.choice(_RELS), name=rng.choice(_NAMES), city=rng.choice(_CITIES), job=rng.choice(_JOBS),
            food=rng.choice(_FOODS), pet=rng.choice(_PETS), petname=rng.choice(_PET_NAMES), num=rng.randint(2, 45),
            hobby=rng.choice(_HOBBIES), weekday=rng.choice(_WEEKDAYS), company=rng.choice(_COMPANIES),
            brand=rng.choice(_BRANDS), color=rng.choice(_COLORS), band=rng.choice(_BANDS), month=rng.choice(_MONTHS),
            day=rng.randint(1, 28), big=rng.randint(5, 300), major=rng.choice(_MAJORS), weather=rng.choice(_WEATHER),
            drink=rng.choice(_DRINKS), lang=rng.choice(_LANGS), gadget=rng.choice(_GADGETS), genre=rng.choice(_GENRES))
        if n > 2000:  # beyond the template space, add a harmless suffix to keep texts unique
            s += rng.choice(["", "，你記一下", "，之前好像說過", "，哈哈", "，對了", "，先這樣", "，真的"])
            if s in seen:
                s += f"（{rng.randint(1, 10**6)}）"
        if s in seen:
            continue
        seen.add(s)
        out.append(s)
    return out


def documents(n_distractors: int) -> List[str]:
    """All documents: the labelled facts first (ids 0..34), then distractors."""
    return FACTS + distractors(n_distractors)
