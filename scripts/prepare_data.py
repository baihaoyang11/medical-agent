"""数据准备与就绪检查 —— 不依赖任何第三方库，不需要 API Key。

    python scripts/prepare_data.py           # 等价于 --check
    python scripts/prepare_data.py --check   # 检查 data/ 与 pdf_output/ 是否就绪
    python scripts/prepare_data.py --demo    # 生成 data/demo.jsonl（24 条演示问答）

为什么需要这个脚本：
    新克隆仓库的人第一件会撞上的墙不是代码，是"数据在哪"。
    仓库不放原始语料（体积大、有版权），所以必须有一个脚本明确告诉
    使用者"缺什么、放哪里、没有的话怎么先用演示数据跑通"。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
PDF_DIR = ROOT / "pdf_output"

REQUIRED_KEYS = ("query", "response")


# ────────────────────────────────────────────────────────────────
# 演示数据：12 组，覆盖饮食 / 用药 / 鉴别 / 急症前兆等典型问法
# ────────────────────────────────────────────────────────────────
DEMO_PAIRS = [
    (
        "糖尿病患者的饮食禁忌有哪些",
        "糖尿病患者应控制总热量与碳水化合物比例，主食可用杂粮替代精制米面，"
        "避免含糖饮料、糕点和高糖水果过量摄入。增加蔬菜、豆类和膳食纤维，"
        "规律三餐并监测餐后血糖。合并高血压者每日食盐不超过五克，"
        "合并肾病者需在医生指导下限制蛋白质总量。戒烟限酒，避免空腹饮酒引起低血糖。",
    ),
    (
        "高血压需要终身服药吗",
        "原发性高血压目前无法根治，多数患者需要长期甚至终身服药控制血压。"
        "擅自停药会导致血压反弹，增加脑卒中、心肌梗死和肾损害风险。"
        "部分轻度患者在改善生活方式后可在医生评估下减量，但必须在监测下进行，"
        "不能自行停药。常用药物包括钙通道阻滞剂、ACEI、ARB 和利尿剂，"
        "具体方案需结合合并症由医生制定。",
    ),
    (
        "普通感冒和流感怎么区分",
        "普通感冒多由鼻病毒引起，症状以鼻塞、流涕、咽痛为主，全身症状轻，"
        "一般三到七天自愈。流感由流感病毒引起，起病急，高热可达三十九度以上，"
        "伴明显头痛、全身肌肉酸痛和乏力，病程更长，容易并发肺炎。"
        "流感在流行季节有聚集性发病特点，确诊需病原学检测。"
        "高危人群出现流感症状应尽早就医并在四十八小时内开始抗病毒治疗。",
    ),
    (
        "发烧三十八度五需要吃退烧药吗",
        "发热是机体对抗感染的防御反应，三十八度五以下且精神状态良好时，"
        "可先物理降温、多饮水并观察。超过三十八度五或明显不适时可用对乙酰氨基酚"
        "或布洛芬，两次用药间隔至少四到六小时，二十四小时内不超过四次。"
        "儿童禁用阿司匹林。持续发热超过三天、伴意识改变、抽搐、呼吸困难或皮疹者，"
        "应立即就医而不是继续自行退热。",
    ),
    (
        "手足口病怎么预防",
        "手足口病由肠道病毒引起，多见于五岁以下儿童，通过消化道、呼吸道飞沫"
        "和接触疱疹液传播。预防重点是勤洗手，尤其是饭前便后和外出回家后；"
        "玩具餐具定期消毒；流行季节避免去人群密集场所。"
        "EV71 疫苗可预防由 EV71 引起的重症手足口病，建议六月龄至五岁儿童接种。"
        "患儿应居家隔离至症状消失后一周，密切观察有无持续高热、肢体抖动等重症信号。",
    ),
    (
        "孕妇为什么要补叶酸",
        "叶酸参与一碳代谢和 DNA 合成，孕早期缺乏会显著增加胎儿神经管缺陷风险。"
        "建议从备孕期开始每日补充四百微克，至少持续到孕满三个月。"
        "既往生育过神经管缺陷患儿或服用抗癫痫药物的女性，需在医生指导下加量。"
        "同时可多吃深绿色蔬菜、豆类和动物肝脏，但食物叶酸吸收率有限，"
        "不能替代补充剂。",
    ),
    (
        "痛风患者不能吃哪些食物",
        "痛风患者应限制高嘌呤食物，包括动物内脏、浓肉汤、沙丁鱼凤尾鱼等海鱼、"
        "贝类以及啤酒。红肉和含果糖饮料也应控制。"
        "急性发作期严格限嘌呤，缓解期可适量摄入中等嘌呤食物。"
        "每日饮水两千毫升以上促进尿酸排泄，避免饥饿性酮症和剧烈运动诱发发作。"
        "饮食控制不能替代降尿酸药物，血尿酸持续偏高者需规律用药并监测。",
    ),
    (
        "幽门螺杆菌阳性一定要治疗吗",
        "幽门螺杆菌是胃炎、消化性溃疡和胃癌的明确危险因素。"
        "检测阳性且有消化道症状、溃疡病史、胃癌家族史或长期服用非甾体抗炎药者，"
        "建议根除治疗。目前推荐含铋剂四联方案，疗程十到十四天，"
        "需足量足疗程服药，不可自行停药。治疗结束至少四周后复查呼气试验确认根除，"
        "避免在复查前服用抑酸药影响结果。",
    ),
    (
        "湿疹和牛皮癣有什么区别",
        "湿疹是皮肤屏障功能异常导致的炎症，表现为红斑、丘疹、渗出和剧烈瘙痒，"
        "多对称分布，与过敏和干燥相关。牛皮癣即银屑病，是免疫介导的慢性病，"
        "典型为边界清楚的红斑覆盖银白色鳞屑，刮除鳞屑可见薄膜现象和点状出血，"
        "好发于头皮、肘膝伸侧。两者治疗方向不同，湿疹以保湿和外用激素为主，"
        "银屑病需系统评估，建议皮肤科明确诊断后用药。",
    ),
    (
        "心肌梗死发作前有什么征兆",
        "心肌梗死常见前驱症状是原本稳定的心绞痛变得频繁、程度加重、"
        "持续时间延长，含服硝酸甘油缓解变差。发作时典型表现为胸骨后压榨样疼痛，"
        "可放射至左肩、左臂内侧、下颌或上腹，伴大汗、恶心和濒死感，"
        "持续时间超过二十分钟。老年人、女性和糖尿病患者可能仅表现为乏力、"
        "气促或上腹不适。出现上述情况应立即拨打急救电话，不要自行驾车前往。",
    ),
    (
        "抗生素滥用有什么危害",
        "抗生素只对细菌感染有效，对普通感冒等病毒感染无效。"
        "滥用会筛选出耐药菌，导致后续感染无药可用，同时破坏肠道菌群，"
        "引起腹泻和真菌感染，还可能带来肝肾损害和过敏反应。"
        "使用抗生素应遵循医嘱，足量足疗程，不自行购买、不随意停药、"
        "不与人共用剩余药物。养殖业抗生素残留也是耐药菌的重要来源。",
    ),
    (
        "流感疫苗每年都要打吗",
        "需要。流感病毒抗原变异快，世界卫生组织每年根据全球监测结果更新疫苗组分，"
        "往年接种产生的抗体对新流行株保护力下降。接种后抗体维持约六到八个月，"
        "因此建议每年流感季前接种一次。优先推荐人群包括六十岁以上老人、"
        "六个月至五岁儿童、孕妇、慢性病患者和医务人员。"
        "对疫苗成分严重过敏者禁忌接种。",
    ),
]


def _fail(msg: str) -> None:
    print(f"[FAIL] {msg}")


def _ok(msg: str) -> None:
    print(f"[OK]   {msg}")


def _warn(msg: str) -> None:
    print(f"[WARN] {msg}")


def check_jsonl(path: Path) -> tuple[int, int]:
    """返回 (行数, 有效行数)。"""
    total = valid = 0
    with path.open("r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            total += 1
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                _warn(f"{path.name} 第 {lineno} 行不是合法 JSON：{exc}")
                continue
            missing = [k for k in REQUIRED_KEYS if not obj.get(k)]
            if missing:
                _warn(f"{path.name} 第 {lineno} 行缺字段：{', '.join(missing)}")
                continue
            valid += 1
    return total, valid


def do_check() -> int:
    print("== 数据就绪检查 ==\n")
    problems = 0

    # 1. QA 语料
    candidates = [DATA_DIR / "data.jsonl", DATA_DIR / "demo.jsonl"]
    found = next((p for p in candidates if p.exists()), None)

    if found is None:
        _fail("找不到 QA 语料：data/data.jsonl 与 data/demo.jsonl 都不存在")
        print("       解决：python scripts/prepare_data.py --demo   # 先生成演示数据")
        print("       或参考 data/README.md 放置真实语料")
        problems += 1
    else:
        total, valid = check_jsonl(found)
        if valid == 0:
            _fail(f"{found.name} 有 {total} 行，但没有一行同时包含 query 与 response")
            problems += 1
        elif valid < total:
            _warn(f"{found.name}：共 {total} 行，有效 {valid} 行（其余行被跳过）")
            _ok(f"{found.name} 可用于入库：{valid} 条")
        else:
            _ok(f"{found.name}：{valid} 条问答对，格式正确")

    # 2. PDF 结构化结果
    xlsx = PDF_DIR / "pdf_detailed_text.xlsx"
    if xlsx.exists():
        size_mb = xlsx.stat().st_size / 1024 / 1024
        _ok(f"PDF 结构化数据存在：pdf_output/pdf_detailed_text.xlsx（{size_mb:.1f} MB）")
    else:
        _warn("pdf_output/pdf_detailed_text.xlsx 不存在 —— PDF 那一路检索会没有数据")
        print("       解决：python preprocess.py   # 需要先把教材 PDF 放进 pdf_documents/")

    # 3. 向量库
    print()
    try:
        sys.path.insert(0, str(ROOT))
        from config import config  # noqa: PLC0415

        for label, uri in (("QA 库", config.MILVUS_QA_URI), ("PDF 库", config.MILVUS_PDF_URI)):
            p = Path(uri)
            if p.exists():
                _ok(f"{label}已存在：{uri}")
            else:
                _warn(f"{label}不存在：{uri}（首次运行 python vectors1.py 会创建）")
    except Exception as exc:  # noqa: BLE001
        _warn(f"跳过向量库检查（无法加载 config：{exc}）")

    print()
    if problems:
        print(f"结论：{problems} 项阻塞，按上面提示处理后重跑本命令。")
        return 1
    print("结论：数据就绪，可以执行 python vectors1.py 建库。")
    return 0


def do_demo() -> int:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    out = DATA_DIR / "demo.jsonl"
    with out.open("w", encoding="utf-8") as fh:
        for q, a in DEMO_PAIRS:
            fh.write(json.dumps({"query": q, "response": a}, ensure_ascii=False) + "\n")

    avg = sum(len(a) for _, a in DEMO_PAIRS) // len(DEMO_PAIRS)
    _ok(f"已生成 {out.relative_to(ROOT)}：{len(DEMO_PAIRS)} 条问答对（平均回答 {avg} 字）")
    print("\n用途：验证建库与检索链路是否通，不代表真实效果。")
    print("下一步：python vectors1.py   # 注意：需要 ZHIPU_API_KEY")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="数据准备与就绪检查")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--check", action="store_true", help="检查数据是否就绪（默认）")
    group.add_argument("--demo", action="store_true", help="生成 data/demo.jsonl 演示数据")
    args = parser.parse_args()

    code = do_demo() if args.demo else do_check()
    raise SystemExit(code)


if __name__ == "__main__":
    main()
