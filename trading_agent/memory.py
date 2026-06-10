"""
交易记忆系统

功能：
1. 保存交易教训（文本 + 向量 + 标签）
2. 根据当前市场情况搜索最相关的历史教训
3. 让 AI 在交易决策前自动回顾相关教训

工作原理：
- 每条教训存入时用 fastembed 转成向量
- 搜索时把当前场景也转成向量，找最像的 3 条
- 把教训放进 prompt 里，AI 决策前就能看到
"""

import json
import os
import datetime
import numpy as np

_LESSONS_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "lessons.json")

# ---------- 向量模型 ----------
_embedding_model = None


def _get_model():
    """懒加载 embedding 模型（第一次用时才加载）"""
    global _embedding_model
    if _embedding_model is None:
        try:
            from fastembed import TextEmbedding
            # 中文 embedding 模型，轻量~30MB
            _embedding_model = TextEmbedding("BAAI/bge-small-zh-v1.5")
        except Exception as e:
            print(f"  [记忆] 模型加载失败（不影响核心功能）: {e}")
            return None
    return _embedding_model


def _ensure_file():
    """确保 lessons.json 存在"""
    os.makedirs(os.path.dirname(_LESSONS_FILE), exist_ok=True)
    if not os.path.exists(_LESSONS_FILE):
        with open(_LESSONS_FILE, "w") as f:
            json.dump([], f, ensure_ascii=False, indent=2)


def _cosine_similarity(a, b):
    """计算两个向量之间的余弦相似度"""
    a = np.array(a)
    b = np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))


# ========== 公开接口 ==========

def add_lesson(lesson_text: str, tags: list[str] = None, outcome: str = ""):
    """
    添加一条交易教训
    lesson_text: 教训内容，如 "BTC 在30分钟内涨幅超3%时追多容易亏"
    tags: 标签，如 ["btc", "追多", "fomo"]
    outcome: 结果，如 "亏损2%"
    """
    _ensure_file()
    with open(_LESSONS_FILE) as f:
        lessons = json.load(f)

    # 生成向量
    model = _get_model()
    embedding = []
    if model:
        try:
            # fastembed 返回生成器
            for e in model.embed(lesson_text):
                embedding = e.tolist()
                break
        except Exception:
            pass

    lesson = {
        "id": len(lessons) + 1,
        "date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
        "lesson": lesson_text,
        "tags": tags or [],
        "outcome": outcome,
        "embedding": embedding,
        "used_count": 0,
    }
    lessons.append(lesson)
    with open(_LESSONS_FILE, "w") as f:
        json.dump(lessons, f, ensure_ascii=False, indent=2)

    print(f"  [记忆] 保存教训 #{lesson['id']}: {lesson_text[:50]}...")
    return lesson


def recall(context_text: str, top_k: int = 3):
    """
    搜索与当前场景最相关的历史教训
    context_text: 当前市场情况的描述
    top_k: 返回前几条
    返回：最相关的教训列表，每条包含 lesson, tags, outcome
    """
    _ensure_file()
    with open(_LESSONS_FILE) as f:
        lessons = json.load(f)

    if not lessons:
        return []

    # 如果 lessons 没有 embedding（旧数据），返回最近几条
    if not lessons[0].get("embedding"):
        recent = lessons[-top_k:]
        for l in recent:
            l["used_count"] = l.get("used_count", 0) + 1
        with open(_LESSONS_FILE, "w") as f:
            json.dump(lessons, f, ensure_ascii=False, indent=2)
        return [
            {"lesson": l["lesson"], "tags": l.get("tags", []), "outcome": l.get("outcome", "")}
            for l in recent
        ]

    # 给 context 生成向量
    model = _get_model()
    if model:
        try:
            ctx_vec = None
            for e in model.embed(context_text):
                ctx_vec = e
                break
            if ctx_vec is not None:
                # 算相似度，排序
                scored = []
                for l in lessons:
                    if l.get("embedding"):
                        sim = _cosine_similarity(ctx_vec, l["embedding"])
                        scored.append((sim, l))
                scored.sort(key=lambda x: -x[0])
                top = scored[:top_k]
                for _, l in top:
                    l["used_count"] = l.get("used_count", 0) + 1
                with open(_LESSONS_FILE, "w") as f:
                    json.dump(lessons, f, ensure_ascii=False, indent=2)
                return [
                    {"lesson": l["lesson"], "tags": l.get("tags", []), "outcome": l.get("outcome", ""),
                     "similarity": round(s, 3)}
                    for s, l in top
                ]
        except Exception as e:
            print(f"  [记忆] 向量搜索失败: {e}")

    # 向量搜索失败 → 按标签匹配降级
    keywords = context_text.lower().split()
    scored = []
    for l in lessons:
        score = 0
        for tag in l.get("tags", []):
            if tag.lower() in context_text.lower():
                score += 1
        scored.append((score, l))
    scored.sort(key=lambda x: -x[0])
    return [
        {"lesson": l["lesson"], "tags": l.get("tags", []), "outcome": l.get("outcome", "")}
        for _, l in scored[:top_k]
    ]


def get_all_lessons():
    """获取所有教训（不含 embedding）"""
    _ensure_file()
    with open(_LESSONS_FILE) as f:
        lessons = json.load(f)
    return [
        {"id": l["id"], "date": l["date"], "lesson": l["lesson"],
         "tags": l.get("tags", []), "outcome": l.get("outcome", ""),
         "used_count": l.get("used_count", 0)}
        for l in lessons
    ]


def delete_lesson(lesson_id: int):
    """删除一条教训"""
    _ensure_file()
    with open(_LESSONS_FILE) as f:
        lessons = json.load(f)
    lessons = [l for l in lessons if l["id"] != lesson_id]
    with open(_LESSONS_FILE, "w") as f:
        json.dump(lessons, f, ensure_ascii=False, indent=2)
