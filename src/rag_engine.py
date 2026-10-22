"""
RAG Engine for CST Cloud Computing Regulatory Framework (RS10)
Integrates Hybrid Retrieval, Regulatory Knowledge Base, and Generation (Gemini LLM / Expert Fallback).
"""

import os
import json
from typing import Dict, Any, List, Optional
from dotenv import load_dotenv

from src.retriever import HybridRetriever
from src.storage_rules import StorageRulesEngine

load_dotenv()

class CloudRegulationsRAG:
    """
    Main RAG orchestrator for querying CST Cloud Computing Regulations.
    """

    def __init__(self, data_path: Optional[str] = None, api_key: Optional[str] = None):
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.chunks_path = data_path or os.path.join(base_dir, "data", "cst_chunks.json")
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        self.retriever = HybridRetriever(self.chunks_path, self.api_key)
        self.rules_engine = StorageRulesEngine()
        self._init_llm()

    def _init_llm(self):
        """Initializes Google Generative AI client if key is present."""
        self.model = None
        if self.api_key:
            try:
                import google.generativeai as genai
                genai.configure(api_key=self.api_key)
                # Use fast, modern Gemini model
                self.model = genai.GenerativeModel("gemini-1.5-flash")
            except Exception as e:
                print(f"[Warning] Failed to initialize Gemini model: {e}")
                self.model = None

    def update_api_key(self, api_key: str):
        """Allows dynamic API key update at runtime."""
        self.api_key = api_key
        os.environ["GEMINI_API_KEY"] = api_key
        self._init_llm()

    def query(self, user_query: str, top_k: int = 3) -> Dict[str, Any]:
        """
        Executes end-to-end RAG pipeline:
        1. Query analysis & rule matching
        2. Chunk retrieval from CST document
        3. Augmented response generation
        """
        if not user_query or not user_query.strip():
            return {
                "error": "الرجاء إدخال نوع البيانات أو السؤال للاستفسار عن موقع التخزين."
            }

        # Step 1: Retrieval
        retrieved_results = self.retriever.retrieve(user_query, top_k=top_k)
        retrieved_chunks = [r["chunk"] for r in retrieved_results]

        # Step 2: Regulatory Analysis (Deterministic Truth Grounding)
        analysis = self.rules_engine.analyze_data_type(user_query)
        rule = analysis["rule"]

        # Step 3: Generation (Gemini LLM or Grounded Expert Generator)
        if self.model:
            answer_text, gen_mode = self._generate_with_gemini(user_query, retrieved_chunks, rule)
        else:
            answer_text, gen_mode = self._generate_grounded_fallback(user_query, retrieved_chunks, rule)

        # Assemble full structured response
        return {
            "query": user_query,
            "classification": rule["name_ar"],
            "entity_type": rule["entity"],
            "storage_location": rule["storage_location"],
            "provider_class": rule["provider_class"],
            "cross_border_policy": rule["cross_border"],
            "approval_required": rule["approval_required"],
            "legal_articles": rule["articles"],
            "classification_criteria": rule["criteria"],
            "answer": answer_text,
            "generation_mode": gen_mode,
            "retrieved_chunks": [
                {
                    "title": c.get("title"),
                    "page": c.get("page"),
                    "section_id": c.get("section_id"),
                    "score": r["score"],
                    "content": c.get("content")
                }
                for r, c in zip(retrieved_results, retrieved_chunks)
            ]
        }

    def _generate_with_gemini(self, user_query: str, chunks: List[Dict[str, Any]], rule: Dict[str, Any]) -> tuple[str, str]:
        """Generates answer using Gemini model augmented with regulatory context."""
        context_str = "\n\n".join([
            f"--- وثيقة تنظيمات الحوسبة السحابية RS10 (صفحة {c.get('page')}, مادة {c.get('section_id')}) ---\n"
            f"العنوان: {c.get('title')}\n"
            f"المحتوى: {c.get('content')}"
            for c in chunks
        ])

        system_prompt = f"""أنت مستشار قانوني وتنظيمي خبير في أنظمة الحوسبة السحابية وهيئة الاتصالات والفضاء والتقنية (CST) في المملكة العربية السعودية.
المطلوب منك الإجابة بدقة وبأسلوب رسمي وواضح ومفصل على سؤال المستخدم حول نوع البيانات ومكان تخزينها المناسب والشروط النظامية المرتبطة بها.

اعتمد حصراً على النصوص والضوابط الواردة في السياق المرفق من وثيقة "تنظيمات تقديم خدمات الحوسبة السحابية RS10 (الإصدار الرابع - أكتوبر 2023م)":

الحقائق التنظيمية الأساسية المعتمدة لهذا التصنيف:
- مستوى التصنيف: {rule['name_ar']} ({rule['entity']})
- موقع التخزين الإلزامي: {rule['storage_location']}
- فئة مقدم الخدمة المؤهل: {rule['provider_class']}
- سياسة النقل عبر الحدود: {rule['cross_border']}
- الموافقات المطلوبة: {rule['approval_required']}
- المواد المستند إليها: {", ".join(rule['articles'])}

السياق المسترجع من وثيقة التنظيمات:
{context_str}

تعليمات الرد:
1. اذكر بوضوح وبشكل مباشر: أين يجب أن تُخزّن هذه البيانات؟ (داخل المملكة حصراً أو يجوز خارجها).
2. فئة مقدم الخدمة السحابية المؤهل للتعامل معها (الفئة أ، ب، ج) بحسب الجدول رقم 2.
3. الاشتراطات الإلزامية والموافقات المطلوبة (مثل موافقة الهيئة CST المسبقة للبيانات السرية، أو موافقة المشترك المسبقة للبيانات غير الحكومية).
4. حكم النقل للخارج والنسخ الاحتياطي بموجب المادة 6-3-3.
5. استشهد بالمواد والجداول ورقم الصفحة في الوثيقة لدعم الإجابة.
"""

        try:
            response = self.model.generate_content([
                system_prompt,
                f"سؤال المستخدم: {user_query}"
            ])
            return response.text.strip(), "Gemini LLM (Augmented)"
        except Exception as e:
            print(f"[Warning] Gemini generation failed: {e}. Falling back to expert rule generator.")
            return self._generate_grounded_fallback(user_query, chunks, rule)

    def _generate_grounded_fallback(self, user_query: str, chunks: List[Dict[str, Any]], rule: Dict[str, Any]) -> tuple[str, str]:
        """
        Synthesizes a complete, highly-grounded regulatory consultation
        matching the CST Cloud Regulations RS10 without requiring external API calls.
        """
        articles_str = "، ".join(rule["articles"])
        
        # Build comprehensive structured Arabic explanation
        explanation = f"""### تقرير التخزين والتصنيف النظامي وفق تنظيمات هيئة الاتصالات والفضاء والتقنية (CST RS10)

بناءً على وثيقة **"تنظيمات تقديم خدمات الحوسبة السحابية (RS10 - الإصدار الرابع)"** الصادرة عن هيئة الاتصالات والفضاء والتقنية:

#### 1. تصنيف البيانات المحددة:
* **مستوى التصنيف**: **{rule['name_ar']}** ({rule['entity']}).
* **معيار التصنيف**: {rule['criteria']}

#### 2. أين يجب أن تُخزن البيانات؟
* **موقع التخزين الإلزامي**: **{rule['storage_location']}**.
* **سياسة النقل عبر الحدود والنسخ الاحتياطي**: {rule['cross_border']}.
  * *تنص المادة (6-3-3)* على إلزام مقدمي الخدمة والمشتركين بعدم نقل أي محتوى من بيانات جهات حكومية سعودية خارج المملكة لأي غرض أو صيغة (سواء كان دائماً أو مؤقتاً كالتخزين المؤقت والاحتياطي) ما لم يسمح به نص صريح في أنظمة المملكة.

#### 3. فئة مقدم خدمة الحوسبة السحابية المؤهل (CSP):
* **فئة التسجيل المطلوبة**: **{rule['provider_class']}** (وفقاً للجدول رقم 2: فئات تسجيل مقدمي الخدمة).
* يُلزم المشترك بالتعاقد مع مقدم خدمة حوسبة سحابية مسجل تسجيلاً صحيحاً وكاملاً لدى الهيئة بموجب المادة (4-3-3) والمادة (7-3-3).

#### 4. الاشتراطات والموافقات النظامية المطلوبة:
* **الموافقات**: {rule['approval_required']}.
* **الالتزام بالجهات الوطنية**: يجب التقيد بالضوابط والسياسات الصادرة عن الهيئة الوطنية للأمن السيبراني (NCA) ومكتب إدارة البيانات الوطنية (NDMO) بموجب المادة (1-11-3).
* **حماية وسرية البيانات**: يُحظر على مقدم الخدمة فك تشفير البيانات أو إتاحتها لأي طرف ثالث دون موافقة خطية صريحة ومسبقة من الهيئة (المادة 2-4-3).

#### 5. الأسانيد والمراجع النظامية من الوثيقة:
* تم الاستناد إلى: **{articles_str}**.
"""
        return explanation.strip(), "نظام التحليل التنظيمي المرجعي (CST RS10 Grounded Engine)"
