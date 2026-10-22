#!/usr/bin/env python3
"""
CLI Tool for CST Cloud Computing Regulations RAG System
Author: SDAIA Agentic AI Program
"""

import sys
import os

# Ensure UTF-8 output on Windows console
if sys.platform.startswith("win"):
    sys.stdout.reconfigure(encoding="utf-8")

from src.rag_engine import CloudRegulationsRAG

def print_banner():
    print("""
======================================================================
  🇸🇦 نظام الاستعلام والامتثال لتنظيمات الحوسبة السحابية (CST RS10)
  مستشار تصنيف وموقع تخزين البيانات السحابية - الإصدار الرابع
======================================================================
    """)

def format_card(res: dict):
    print("\n" + "="*65)
    print("📋 نتيجة التحليل التنظيمي وموقع التخزين الإلزامي")
    print("="*65)
    print(f"🔹 مستوى التصنيف:          {res.get('classification')} ({res.get('entity_type')})")
    print(f"📍 أين يجب أن تخزن؟:        {res.get('storage_location')}")
    print(f"🏢 فئة مقدم الخدمة السحابي: {res.get('provider_class')}")
    print(f"✈️  النقل والنسخ خارج المملكة: {res.get('cross_border_policy')}")
    print(f"📝 الموافقات المطلوبة:      {res.get('approval_required')}")
    print(f"⚖️  الأسانيد النظامية:        {', '.join(res.get('legal_articles', []))}")
    print(f"⚙️  آلية التوليد:            {res.get('generation_mode')}")
    print("-"*65)
    print("📖 الشرح والتفصيل التنظيمي:")
    print(res.get('answer'))
    print("="*65)
    print("📑 المقاطع المسترجعة من الوثيقة الرسمية:")
    for i, chunk in enumerate(res.get('retrieved_chunks', []), 1):
        print(f"  [{i}] صفحة {chunk['page']} | مادة: {chunk['section_id']} | {chunk['title']} (درجة المطابقة: {chunk['score']})")
    print("="*65 + "\n")

def main():
    print_banner()
    rag = CloudRegulationsRAG()
    
    print("اكتب نوع البيانات أو سؤالك (أو اكتب 'خروج' للإنهاء):\n")
    print("أمثلة يمكنك تجربتها:")
    print(" 1. بيانات سرية للغاية تابعة لوزارة حكومية")
    print(" 2. بيانات مقيدة لجهة حكومية")
    print(" 3. بيانات عامة لهيئة حكومية")
    print(" 4. بيانات عملاء شركة خاصة أو متجر إلكتروني هل تخزن خارج المملكة؟")
    print(" 5. بيانات واردة من جهة حكومية إلى مقاول قطاع خاص\n")

    while True:
        try:
            query = input("❓ استفسارك > ").strip()
            if not query:
                continue
            if query in ["خروج", "exit", "quit", "q"]:
                print("مع السلامة! بالتوفيق.")
                break
            
            res = rag.query(query)
            format_card(res)
        except (KeyboardInterrupt, EOFError):
            print("\nتم إنهاء البرنامج.")
            break
        except Exception as e:
            print(f"حدث خطأ أثناء المعالجة: {e}")

if __name__ == "__main__":
    main()
