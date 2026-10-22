import sys
sys.stdout.reconfigure(encoding='utf-8')
from src.rag_engine import CloudRegulationsRAG

rag = CloudRegulationsRAG()

test_queries = [
    "بيانات سرية للغاية لجهة حكومية",
    "بيانات مقيدة لوزارة التعليم",
    "بيانات عامة لهيئة حكومية",
    "بيانات عملاء متجر قطاع خاص هل يجوز تخزينها خارج السعودية؟",
    "بيانات واردة من جهة حكومية لشركة مقاولات",
    "ما هي اشتراطات الفئة ج لمقدمي الخدمات السحابية؟"
]

print("="*60)
print("اختبار نظام RAG لتنظيمات الحوسبة السحابية (CST RS10)")
print("="*60)

for q in test_queries:
    res = rag.query(q)
    print(f"\n[السؤال]: {q}")
    print(f"  * التصنيف: {res.get('classification')}")
    print(f"  * أين تخزن؟: {res.get('storage_location')}")
    print(f"  * فئة مقدم الخدمة: {res.get('provider_class')}")
    print(f"  * سياسة النقل للخارج: {res.get('cross_border_policy')}")
    print(f"  * الموافقات: {res.get('approval_required')}")
    print(f"  * الأسانيد النظامية: {', '.join(res.get('legal_articles', []))}")
    print(f"  * وضع التوليد: {res.get('generation_mode')}")
    print(f"  * المقاطع المسترجعة: {len(res.get('retrieved_chunks', []))} مقاطع")
