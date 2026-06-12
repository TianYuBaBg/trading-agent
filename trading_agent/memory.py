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
import heapq
import threading
import numpy as np

_LESSONS_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "lessons.json")

# 内存缓存，避免同一个 cycle 内重复读文件
_lessons_cache = None

# 线程锁，防止 auto 模式下主线程和输入线程同时写文件
_io_lock = threading.Lock()

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


def _load_lessons():
    """读取 lessons.json，确保文件有效且为 list，返回 lessons 列表（内存缓存，线程安全）"""
    global _lessons_cache
    if _lessons_cache is not None:
        return _lessons_cache
    with _io_lock:
        # 双检锁：获取锁后再检查一次，防止并发初始加载
        if _lessons_cache is not None:
            return _lessons_cache
        os.makedirs(os.path.dirname(_LESSONS_FILE), exist_ok=True)
        if not os.path.exists(_LESSONS_FILE):
            with open(_LESSONS_FILE, "w", encoding="utf-8") as f:
                json.dump([], f, ensure_ascii=False, indent=2)
            _lessons_cache = []
            return []
        try:
            with open(_LESSONS_FILE, encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                _lessons_cache = [l for l in data if isinstance(l, dict)]
                return _lessons_cache
        except (json.JSONDecodeError, ValueError):
            pass
        # 文件损坏或非 list，备份并重建
        backup = _LESSONS_FILE + ".bak"
        try:
            shutil.copy2(_LESSONS_FILE, backup)
        except Exception:
            pass
        print(f"  [记忆] lessons.json 损坏，已备份到 {backup}，重新创建")
        with open(_LESSONS_FILE, "w", encoding="utf-8") as f:
            json.dump([], f, ensure_ascii=False, indent=2)
        _lessons_cache = []
        return []


def _cosine_similarity(a, b):
    """计算两个向量之间的余弦相似度"""
    a = np.array(a)
    b = np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))


def _increment_used(ids_to_inc, lessons=None):
    """将指定 id 的教训 used_count 加 1，写回文件
    lessons: 可选，已加载的教训列表，避免重复读取文件"""
    if not ids_to_inc:
        return
    if lessons is None:
        lessons = _load_lessons()
    with _io_lock:
        for l in lessons:
            lid = l.get("id")
            if lid is not None and lid in ids_to_inc:
                l["used_count"] = (l.get("used_count") or 0) + 1
        with open(_LESSONS_FILE, "w", encoding="utf-8") as f:
            json.dump(lessons, f, ensure_ascii=False, indent=2)


def _keyword_score(lesson: dict, context_text: str, ctx_chinese: set) -> float:
    """计算一条教训与上下文文本的关键词匹配分数"""
    score = 0
    lesson_lower = (lesson.get("lesson") or "").lower()
    ctx_lower = context_text.lower()
    for tag in (lesson.get("tags") or []):
        if tag.lower() in ctx_lower:
            score += 2
    for kw in set(ctx_lower.split()):
        if kw in lesson_lower:
            score += 1
    if ctx_chinese:
        lesson_cjk = set(re.findall(r'[一-鿿]', lesson_lower))
        overlap = len(ctx_chinese & lesson_cjk) / max(len(ctx_chinese), 1)
        score += overlap * 5
    return score


# ========== 公开接口 ==========

def add_lesson(lesson_text: str, tags: list[str] = None, outcome: str = ""):
    """
    添加一条交易教训
    lesson_text: 教训内容，如 "BTC 在30分钟内涨幅超3%时追多容易亏"
    tags: 标签，如 ["btc", "追多", "fomo"]
    outcome: 结果，如 "亏损2%"
    """
    if not isinstance(lesson_text, str):
        return None
    lesson_text = lesson_text.strip()
    if len(lesson_text) < 5:
        return None

    # 归一化 tags：允许字符串传入，强制转为字符串列表（小写 + 去重）
    if isinstance(tags, str):
        tags = [t.strip().lower() for t in tags.split(",") if t.strip()]
    elif tags is None:
        tags = []
    else:
        tags = [str(t).strip().lower() for t in tags if t is not None and str(t).strip()]
    tags = list(set(tags))

    # 生成向量（I/O 密集，放锁外避免阻塞其他线程）
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

    # 文件读取 + 去重检查 + 追加写入（原子操作）
    with _io_lock:
        lessons = _load_lessons()
        # 去重：完整文本相同才视为重复
        clean = lesson_text
        for existing in lessons:
            if clean == (existing.get("lesson") or "").strip():
                return existing

        lesson = {
            "id": max(((l.get("id") or 0) for l in lessons), default=0) + 1,
            "date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
            "lesson": lesson_text,
            "tags": tags or [],
            "outcome": outcome or "",
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
    if not isinstance(context_text, str):
        context_text = str(context_text)
    else:
        context_text = context_text or ""
    top_k = max(1, min(int(top_k), 20))
    lessons = _load_lessons()

    if not lessons:
        return []

    # 如果所有 lessons 都没有 embedding（旧数据），按 used_count 升序返回（最少使用的优先）
    has_any_embedding = any(l.get("embedding") for l in lessons)
    if not has_any_embedding:
        # 使用 heapq.nsmallest 获取 used_count 最小的 top_k 条（性能更好）
        selected = heapq.nsmallest(top_k, lessons, key=lambda l: l.get("used_count") or 0)
        _increment_used({l.get("id") for l in selected}, lessons=lessons)
        return [
            {"lesson": (l.get("lesson") or ""), "tags": (l.get("tags") or []), "outcome": (l.get("outcome") or ""),
             "similarity": 0.0}
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
                    ctx_chinese = set(re.findall(r'[一-鿿]', context_text.lower()))
                    remaining = [l for l in lessons if l.get("id") not in selected_ids and not l.get("embedding")]
                    fallback = [(_keyword_score(l, context_text, ctx_chinese), l) for l in remaining]
                    fallback.sort(key=lambda x: -x[0])
                    top.extend((0.0, l) for _, l in fallback[:top_k - len(top)])

                _increment_used({l.get("id") for _, l in top}, lessons=lessons)
                return [
                    {"lesson": (l.get("lesson") or ""), "tags": (l.get("tags") or []), "outcome": (l.get("outcome") or ""),
                     "similarity": round(s, 3)}
                    for s, l in top
                ]
        except Exception as e:
            print(f"  [记忆] 向量搜索失败: {e}")

    # 向量搜索失败 → 关键词匹配降级（搜索标签 + 正文）
    ctx_chinese = set(re.findall(r'[一-鿿]', context_text.lower()))
    scored = [(_keyword_score(l, context_text, ctx_chinese), l) for l in lessons]
    scored.sort(key=lambda x: -x[0])
    top_ids = {l.get("id") for _, l in scored[:top_k]}
    _increment_used(top_ids, lessons=lessons)
    return [
        {"lesson": (l.get("lesson") or ""), "tags": (l.get("tags") or []), "outcome": (l.get("outcome") or ""),
         "similarity": 0.0}
        for _, l in scored[:top_k]
    ]


def get_all_lessons():
    """获取所有教训（不含 embedding）"""
    lessons = _load_lessons()
    return [
        {"id": l.get("id") or i + 1, "date": l.get("date") or "", "lesson": l.get("lesson") or "",
         "tags": (l.get("tags") or []), "outcome": (l.get("outcome") or ""),
         "used_count": l.get("used_count") or 0}
        for i, l in enumerate(lessons)
    ]
