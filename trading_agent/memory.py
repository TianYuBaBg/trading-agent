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
import re
import shutil
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
            _embedding_model = False
            return None
    return _embedding_model if _embedding_model else None


def _ensure_file():
    """确保 lessons.json 存在且内容有效"""
    os.makedirs(os.path.dirname(_LESSONS_FILE), exist_ok=True)
    if not os.path.exists(_LESSONS_FILE):
        with open(_LESSONS_FILE, "w", encoding="utf-8") as f:
            json.dump([], f, ensure_ascii=False, indent=2)
        return
    # 检查现有文件是否损坏
    try:
        with open(_LESSONS_FILE, encoding="utf-8") as f:
            json.load(f)
    except (json.JSONDecodeError, ValueError):
        backup = _LESSONS_FILE + ".bak"
        shutil.copy2(_LESSONS_FILE, backup)
        print(f"  [记忆] lessons.json 损坏，已备份到 {backup}，重新创建")
        with open(_LESSONS_FILE, "w", encoding="utf-8") as f:
            json.dump([], f, ensure_ascii=False, indent=2)


def _cosine_similarity(a, b):
    """计算两个向量之间的余弦相似度"""
    a = np.array(a)
    b = np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))


def _increment_used(ids_to_inc):
    """将指定 id 的教训 used_count 加 1，写回文件"""
    if not ids_to_inc:
        return
    _ensure_file()
    with open(_LESSONS_FILE, encoding="utf-8") as f:
        lessons = json.load(f)
    for l in lessons:
        if l.get("id") in ids_to_inc:
            l["used_count"] = l.get("used_count", 0) + 1
    with open(_LESSONS_FILE, "w", encoding="utf-8") as f:
        json.dump(lessons, f, ensure_ascii=False, indent=2)


# ========== 公开接口 ==========

def add_lesson(lesson_text: str, tags: list[str] = None, outcome: str = ""):
    """
    添加一条交易教训
    lesson_text: 教训内容，如 "BTC 在30分钟内涨幅超3%时追多容易亏"
    tags: 标签，如 ["btc", "追多", "fomo"]
    outcome: 结果，如 "亏损2%"
    """
    _ensure_file()
    with open(_LESSONS_FILE, encoding="utf-8") as f:
        lessons = json.load(f)

    lesson_text = lesson_text.strip()
    if len(lesson_text) < 5:
        return None

    # 归一化 tags：允许字符串传入，强制转为字符串列表
    if isinstance(tags, str):
        tags = [t.strip() for t in tags.split(",") if t.strip()]
    elif tags is None:
        tags = []
    else:
        tags = [str(t).strip() for t in tags if str(t).strip()]

    # 去重：完整文本相同才视为重复（严格匹配，避免误杀相似但不同的教训）
    clean = lesson_text.strip()
    if len(clean) >= 20:  # 降低阈值到 20 字符，更严格地防止重复
        for existing in lessons:
            existing_text = existing.get("lesson", "").strip()
            # 只有当完整文本完全相同时才视为重复
            if clean == existing_text:
                return existing

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
        "id": max((l.get("id", 0) for l in lessons), default=0) + 1,
        "date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
        "lesson": lesson_text,
        "tags": tags or [],
        "outcome": outcome,
        "embedding": embedding,
        "used_count": 0,
    }
    lessons.append(lesson)
    with open(_LESSONS_FILE, "w", encoding="utf-8") as f:
        json.dump(lessons, f, ensure_ascii=False, indent=2)

    print(f"  [记忆] 保存教训 #{lesson['id']}: {lesson_text[:50]}...")
    return lesson


def recall(context_text: str, top_k: int = 3):
    """
    搜索与当前场景最相关的历史教训
    context_text: 当前市场情况的描述
    top_k: 返回前几条（限制在 1~20）
    返回：最相关的教训列表，每条包含 lesson, tags, outcome
    """
    _ensure_file()
    context_text = context_text or ""
    top_k = max(1, min(int(top_k), 20))
    with open(_LESSONS_FILE, encoding="utf-8") as f:
        lessons = json.load(f)

    if not lessons:
        return []

    # 如果所有 lessons 都没有 embedding（旧数据），按 used_count 升序返回（最少使用的优先）
    has_any_embedding = any(l.get("embedding") for l in lessons)
    if not has_any_embedding:
        # 使用 heapq.nsmallest 获取 used_count 最小的 top_k 条（性能更好）
        import heapq
        selected = heapq.nsmallest(top_k, lessons, key=lambda l: l.get("used_count", 0))
        _increment_used({l.get("id") for l in selected})
        return [
            {"lesson": l["lesson"], "tags": l.get("tags", []), "outcome": l.get("outcome", "")}
            for l in selected
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

                # 向量结果不足 top_k 时，用关键词匹配补充无 embedding 的教训
                if len(top) < top_k:
                    selected_ids = {l[1].get("id") for l in top}
                    keywords = set(context_text.lower().split())
                    ctx_chinese = set(re.findall(r'[一-鿿]', context_text.lower()))
                    remaining = [l for l in lessons if l.get("id") not in selected_ids and not l.get("embedding")]
                    fallback = []
                    for l in remaining:
                        score = 0
                        lesson_lower = l.get("lesson", "").lower()
                        for tag in l.get("tags", []):
                            if tag.lower() in context_text.lower():
                                score += 2
                        for kw in keywords:
                            if kw in lesson_lower:
                                score += 1
                        if ctx_chinese:
                            lesson_cjk = set(re.findall(r'[一-鿿]', lesson_lower))
                            overlap = len(ctx_chinese & lesson_cjk) / max(len(ctx_chinese), 1)
                            score += overlap * 5
                        fallback.append((score, l))
                    fallback.sort(key=lambda x: -x[0])
                    top.extend(fallback[:top_k - len(top)])

                _increment_used({l.get("id") for _, l in top})
                return [
                    {"lesson": l["lesson"], "tags": l.get("tags", []), "outcome": l.get("outcome", ""),
                     "similarity": round(s, 3)}
                    for s, l in top
                ]
        except Exception as e:
            print(f"  [记忆] 向量搜索失败: {e}")

    # 向量搜索失败 → 关键词匹配降级（搜索标签 + 正文）
    keywords = set(context_text.lower().split())
    # 中文语境下 split() 效果很差（中文无空格），用字符重叠作为补充
    ctx_chinese = set(re.findall(r'[一-鿿]', context_text.lower()))
    scored = []
    for l in lessons:
        score = 0
        lesson_lower = l.get("lesson", "").lower()
        for tag in l.get("tags", []):
            if tag.lower() in context_text.lower():
                score += 2
        for kw in keywords:
            if kw in lesson_lower:
                score += 1
        # 中文字符重叠度：降级搜索的关键补充
        if ctx_chinese:
            lesson_cjk = set(re.findall(r'[一-鿿]', lesson_lower))
            overlap = len(ctx_chinese & lesson_cjk) / max(len(ctx_chinese), 1)
            score += overlap * 5
        scored.append((score, l))
    scored.sort(key=lambda x: -x[0])
    top_ids = {l.get("id") for _, l in scored[:top_k]}
    _increment_used(top_ids)
    return [
        {"lesson": l["lesson"], "tags": l.get("tags", []), "outcome": l.get("outcome", "")}
        for _, l in scored[:top_k]
    ]


def get_all_lessons():
    """获取所有教训（不含 embedding）"""
    _ensure_file()
    with open(_LESSONS_FILE, encoding="utf-8") as f:
        lessons = json.load(f)
    return [
        {"id": l.get("id", i + 1), "date": l.get("date", ""), "lesson": l.get("lesson", ""),
         "tags": l.get("tags", []), "outcome": l.get("outcome", ""),
         "used_count": l.get("used_count", 0)}
        for i, l in enumerate(lessons)
    ]
