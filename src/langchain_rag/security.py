"""
Security & Guardrails Module for CST Cloud Regulations Conversational RAG
Handles:
1. Prompt Injection & Instruction Override Defense
2. System Prompt & Instruction Leakage Prevention
3. Out-of-Scope (OOS) Domain Detection & Strict Boundary Enforcement
"""

import re
from typing import Dict, Any, Tuple, List

class SecurityGuardrail:
    """
    Multi-layer security shield to protect the Conversational RAG system
    from prompt injection, jailbreaks, instruction overrides, and out-of-scope inquiries.
    """

    # Prompt injection and jailbreak signatures (English and Arabic)
    INJECTION_PATTERNS = [
        # English patterns
        r"ignore\s+(all\s+)?(previous|prior|above)\s+(instructions|prompts|rules|commands)",
        r"disregard\s+(all\s+)?(previous|prior|above)",
        r"you\s+are\s+now\s+(an?\s+)?(unrestricted|dan|jailbroken|godmode|developer mode)",
        r"forget\s+(all\s+)?(your\s+)?(instructions|rules|constraints|prompt)",
        r"(print|show|output|reveal|leak|repeat|display)\s+(your\s+)?(system\s+prompt|initial\s+prompt|developer\s+instructions)",
        r"bypass\s+(all\s+)?(filters|rules|safety)",
        r"act\s+as\s+an?\s+unfiltered",
        r"override\s+(system|rules|instructions)",
        r"do\s+anything\s+now",
        r"<\|im_start\|>",
        r"<\|im_end\|>",
        r"\[INST\]",
        r"\[/INST\]",

        # Arabic patterns
        r"تجاهل\s+(جميع\s+|كل\s+)?(التعليمات|الأوامر|القواعد|التوجيهات)\s*(السابقة|الأولى)?",
        r"انسَ\s+(جميع\s+|كل\s+)?(التعليمات|الأوامر|القواعد|التوجيهات|قيودك)",
        r"تصرف\s+كـ?\s*(نموذج\s+مفتوح|بدون\s+قيود|دان|dan)",
        r"(اطبع|اظهر|اعرض|اكشف|شارك|انسخ)\s+(البرومبت|التعليمات\s+السرية|توجيهات\s+النظام|التعليمات\s+الأولية|system\s*prompt)",
        r"تخطى\s+(كل\s+|جميع\s+)?(الضوابط|القيود|الفلاتر|الأمان)",
        r"أنت\s+الآن\s+(حر|بدون\s+قيود|نموذج\s+مطور|خارج\s+النظام)",
        r"انسخ\s+النص\s+الذي\s+فوق",
        r"اخرج\s+عن\s+النص"
    ]

    # CST Cloud Regulations Core Domain Keywords (in-scope)
    IN_SCOPE_KEYWORDS = [
        "حوسب", "سحاب", "سحابية", "سحابيه", "بيانات", "مشترك", "مقدم", "تخزين", "معالج",
        "مركز بيانات", "مراكز بيانات", "سري", "سريه", "سري للغايه", "سرية للغاية", "مقيد",
        "مقيده", "مقيّد", "مقيّدة", "عام", "عامه", "عامة", "تصنيف", "توطين", "نقل", "خارج المملك",
        "داخل المملك", "فئ", "فئة", "فئات", "ترخيص", "تسجيل", "عقد", "عقود", "اتفاقي", "مستوى الخدمة",
        "sla", "امان", "أمان", "سيبران", "سيبراني", "تشفير", "فك شفر", "اختراق", "انتهاك",
        "تسريب", "cst", "هيئة الاتصالات", "هيئه الاتصالات", "ndmo", "مكتب ادارة البيانات",
        "nca", "الهيئة الوطنية للأمن", "saip", "ملكية فكرية", "saas", "paas", "iaas",
        "محتوى", "مخالف", "شكاوى", "مسؤولية", "استمرارية اعمال", "طوارئ", "كوارث", "ترشيح",
        "جزاءات", "تفتيش", "سحابة عامة", "سحابة خاصة", "سحابة هجينة", "سحابة مجتمعية",
        "حكومي", "حكوميه", "حكومية", "قطاع خاص", "شرك", "شركات", "افراد", "أفراد", "نطاق",
        "نظام", "تنظيم", "تنظيمات", "وثيقة", "rs10", "لائحة", "اللوائح", "مستشار", "سؤال", "استفسار",
        "نعم", "لا", "ماذا", "كيف", "اين", "أين", "متى", "من", "هل", "ماهي", "ما هي", "وضح", "اشرح"
    ]

    # Explicit Out-of-Scope indicators
    OUT_OF_SCOPE_TOPICS = [
        # Cooking / Food
        "طبخ", "وصفة", "طريقة عمل", "كبسة", "بيتزا", "شاورما", "كعك", "مطعم",
        # Sports
        "كرة قدم", "مباراة", "دوري روشن", "كأس العالم", "نادي الهلال", "نادي النصر", "ميسي", "رونالدو",
        # Weather
        "طقس", "درجة الحرارة", "أمطار", "رياح", "عواصف",
        # General Coding / Unrelated Tech
        "كود جافا سكريبت لعمل موقع", "html", "css", "كتابة لعبة", "react", "vue", "angular",
        # Entertainment / Fiction
        "قصيدة", "شعر", "نكتة", "اغنية", "فيلم", "مسلسل",
        # Unrelated medical / politics
        "علاج الصداع", "دواء", "أعراض كورونا", "مرض السكري", "انتخابات أمريكا"
    ]

    @classmethod
    def check_prompt_injection(cls, query: str) -> Tuple[bool, str]:
        """
        Inspects query for malicious prompt injection or override patterns.
        Returns: (is_injection, reason)
        """
        cleaned = query.strip()
        for pattern in cls.INJECTION_PATTERNS:
            if re.search(pattern, cleaned, re.IGNORECASE):
                return True, f"تم اكتشاف نمط تجاوز للتعليمات النظامية (Matched: {pattern})"
        
        # Check for repetitive system prompt leak prompts
        leak_words = ["system prompt", "البرومبت النظامي", "تعليماتك الأساسية", "تعليمات النظام"]
        if any(w in cleaned.lower() for w in leak_words) and any(verb in cleaned for verb in ["ما هو", "ماهو", "اعطني", "اطبع", "اكشف"]):
            return True, "محاولة استخراج التعليمات الداخلية للنظام (System Prompt Extraction Attempt)"

        return False, ""

    @classmethod
    def check_scope(cls, query: str, chat_history_len: int = 0) -> Tuple[bool, str]:
        """
        Validates if query belongs to CST Cloud Computing Regulations domain.
        Returns: (is_in_scope, reason)
        """
        cleaned = query.strip().lower()
        
        # Check explicit out-of-scope triggers
        for oos in cls.OUT_OF_SCOPE_TOPICS:
            if oos in cleaned:
                return False, f"السؤال يتعلق بموضوع خارج النطاق ({oos})."

        # If it's a follow-up in ongoing conversation (e.g., "وماذا عن المقيدة؟", "وهل يجوز ذلك؟")
        if chat_history_len > 0 and len(cleaned.split()) <= 6:
            # Allow short contextual follow-ups when history exists
            return True, "متابعة سياقية مقبولة"

        # Check in-scope keyword matches
        matched_in_scope = [k for k in cls.IN_SCOPE_KEYWORDS if k in cleaned]
        if not matched_in_scope:
            return False, "السؤال لا يتضمن أي إشارة لمواضيع تنظيمات الحوسبة السحابية أو تصنيف وتخزين البيانات."

        return True, "السؤال يقع ضمن نطاق الوثيقة التنظيمية"

    @classmethod
    def sanitize_output(cls, text: str) -> str:
        """
        Ensures the generated output does not inadvertently leak system prompt instructions
        or internal markers.
        """
        forbidden_substrings = [
            "<|im_start|>", "<|im_end|>", "[INST]", "[/INST]",
            "You are an AI assistant", "system prompt:"
        ]
        sanitized = text
        for sub in forbidden_substrings:
            sanitized = sanitized.replace(sub, "")
        return sanitized.strip()
