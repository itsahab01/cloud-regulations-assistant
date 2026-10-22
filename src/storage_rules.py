"""
CST Cloud Regulations (RS10) - Storage & Classification Rules Engine
This module encapsulates the official regulatory logic defined in:
"تنظيمات تقديم خدمات الحوسبة السحابية RS10 - الإصدار الرابع" (هيئة الاتصالات والفضاء والتقنية CST)
"""

from typing import Dict, Any, List, Optional
import re

class StorageRulesEngine:
    """
    Deterministic rule engine mapping data classification levels to
    storage locations, cloud provider registration classes, and regulatory conditions.
    """

    # Official Classification Mapping according to Table 1 & Table 2
    CLASSIFICATIONS = {
        "top_secret": {
            "name_ar": "سري للغاية",
            "entity": "بيانات جهات حكومية سعودية",
            "storage_location": "داخل المملكة العربية السعودية حصراً (في مراكز بيانات محلية)",
            "provider_class": "الفئة (ج) فقط",
            "cross_border": "ممنوع قطعياً (لا دائم ولا مؤقت ولا نسخ احتياطي)",
            "approval_required": "إلزامي: موافقة مسبقة من هيئة الاتصالات والفضاء والتقنية (CST)",
            "articles": ["جدول 1 (ص 9)", "جدول 2 حاشية 2 (ص 11)", "المادة 6-3-3 (ص 12)", "المادة 7-3-3 (ص 12)"],
            "criteria": "إذا كان الوصول غير المصرح به يؤدي إلى ضرر جسيم واستثنائي لا يمكن تداركه على المصالح الوطنية (معاهدات، سمعة، أمن، دفاع، اقتصاد، بنية تحتية)، أو صحة الأفراد وخصوصية كبار المسؤولين."
        },
        "secret": {
            "name_ar": "سري",
            "entity": "بيانات جهات حكومية سعودية",
            "storage_location": "داخل المملكة العربية السعودية حصراً (في مراكز بيانات محلية)",
            "provider_class": "الفئة (ج) فقط",
            "cross_border": "ممنوع قطعياً (لا دائم ولا مؤقت ولا نسخ احتياطي)",
            "approval_required": "إلزامي: موافقة مسبقة من هيئة الاتصالات والفضاء والتقنية (CST)",
            "articles": ["جدول 1 (ص 9)", "جدول 2 حاشية 2 (ص 11)", "المادة 6-3-3 (ص 12)", "المادة 7-3-3 (ص 12)"],
            "criteria": "إذا كان الوصول غير المصرح به يؤدي إلى ضرر جسيم على المصالح الوطنية، أو خسارة مالية تؤدي لعجز أو إفلاس، أو أذى جسيم يؤثر على حياة مجموعة أفراد، أو قضايا كبرى كتمويل الإرهاب."
        },
        "restricted": {
            "name_ar": "مقيّد",
            "entity": "بيانات جهات حكومية سعودية",
            "storage_location": "داخل المملكة العربية السعودية حصراً (في مراكز بيانات محلية)",
            "provider_class": "الفئة (ب) أو الفئة (ج)",
            "cross_border": "ممنوع قطعياً (لا دائم ولا مؤقت ولا نسخ احتياطي)",
            "approval_required": "التعاقد مع مقدم خدمة مسجل تسجيلاً صحيحاً لدى الهيئة",
            "articles": ["جدول 1 (ص 10)", "جدول 2 (ص 11)", "المادة 6-3-3 (ص 12)", "المادة 7-3-3 (ص 12)"],
            "criteria": "إذا كان الوصول غير المصرح به يؤدي إلى تأثير سلبي محدود على عمل الجهات الحكومية أو الأنشطة الاقتصادية أو شخص معين، أو ضرر محدود على الأصول."
        },
        "public_gov": {
            "name_ar": "عام (حكومي)",
            "entity": "بيانات جهات حكومية سعودية",
            "storage_location": "داخل المملكة العربية السعودية حصراً (بموجب المادة 6-3-3 لجميع البيانات الحكومية)",
            "provider_class": "الفئة (أ) أو الفئة (ب) أو الفئة (ج)",
            "cross_border": "ممنوع إلا بنص صريح مسموح به وفقاً لقوانين المملكة",
            "approval_required": "التعاقد مع مقدم خدمة مسجل لدى الهيئة",
            "articles": ["جدول 1 (ص 10)", "جدول 2 (ص 11)", "المادة 6-3-3 (ص 12)", "المادة 4-3-3 (ص 12)"],
            "criteria": "بيانات حكومية لا يترتب على الإفصاح عنها أي ضرر على المصلحة الوطنية أو الجهات أو الأفراد أو البيئة."
        },
        "nongov_from_gov": {
            "name_ar": "بيانات غير حكومية واردة من جهة حكومية",
            "entity": "بيانات جهات غير حكومية (شركات، مقاولين، جامعات)",
            "storage_location": "داخل المملكة العربية السعودية حصراً (تتبع نفس تصنيفها الحكومي الأصلي)",
            "provider_class": "الفئة (ج) للسرية والسرية للغاية، الفئة (ب أو ج) للمقيدة، الفئة (أ أو ب أو ج) للعامة",
            "cross_border": "ممنوع (حسب التصنيف الحكومي المقابل)",
            "approval_required": "الالتزام بقيود التصنيف الحكومي الواردة به وموافقات CST إذا كانت سرية",
            "articles": ["جدول 1 (ص 10)", "جدول 2 (ص 11)", "المادة 6-3-3 (ص 12)"],
            "criteria": "بيانات واردة من جهة حكومية إلى جهة غير حكومية؛ تصنف كما هي واردة من الجهة الحكومية."
        },
        "nongov_private": {
            "name_ar": "بيانات أخرى (قطاع خاص، شركات، أفراد)",
            "entity": "بيانات جهات غير حكومية",
            "storage_location": "الأصل داخل المملكة، ويجوز خارج المملكة بشروط نظامية",
            "provider_class": "الفئة (أ) أو الفئة (ب) أو الفئة (ج)",
            "cross_border": "مسموح بشريطة إبلاغ المشترك مسبقاً وأخذ موافقته الصريحة (المادة 8-3-3)",
            "approval_required": "موافقة المشترك المسبقة، والالتزام بضوابط مكتب إدارة البيانات الوطنية (NDMO) ونظام حماية البيانات الشخصية (PDPL)",
            "articles": ["جدول 1 (ص 10)", "جدول 2 (ص 11)", "المادة 8-3-3 (ص 12)", "المادة 2-3-4-3 (ص 14)", "المادة 1-11-3 (ص 20)"],
            "criteria": "بيانات تخص مؤسسات تجارية أو شركات خاصة أو مستخدمين أفراد غير تابعة لجهات حكومية."
        }
    }

    @classmethod
    def analyze_data_type(cls, query: str) -> Dict[str, Any]:
        """
        Analyzes the user query and identifies the most relevant classification
        and storage directive according to the CST document.
        """
        from src.retriever import normalize_arabic
        norm_q = normalize_arabic(query)

        # Check entity context
        is_from_gov = any(k in norm_q for k in [
            "وارده من جهه حكوميه", "وارد من جهه حكوميه", "من جهه حكوميه",
            "عقد مع جهه حكوميه", "مشروع حكومي", "عقد حكومي"
        ])
        is_gov = any(k in norm_q for k in [
            "حكوم", "وزار", "امار", "بلدي", "عسكر", "امن", "رئاس", "ديوان", "دفاع", "هيئه", "قطاع عام"
        ])
        is_private = any(k in norm_q for k in [
            "خاص", "شرك", "مؤسس", "مستثمر", "فرد", "تجار", "عميل", "متجر", "بنك", "مصرف", "اهلي"
        ])

        # Check classification levels (on normalized query where 'ة' becomes 'ه' and 'أ'/'إ' becomes 'ا')
        # "سريه للغايه" matches "سري للغاية" or "سرية للغاية"
        is_top_secret = any(k in norm_q for k in [
            "سريه للغايه", "سري للغايه", "top secret", "استثنائي", "سيادي"
        ])
        is_secret = (not is_top_secret) and any(k in norm_q for k in [
            "سري", "سريه", "secret", "تمويل ارهاب", "ارهاب"
        ])
        is_restricted = any(k in norm_q for k in [
            "مقيد", "مقيده", "محدود", "restricted"
        ])
        is_public = any(k in norm_q for k in [
            "عام", "عامه", "مفتوح", "مفتوحه", "public", "للعموم"
        ])

        # Resolution logic
        if is_from_gov:
            res_key = "nongov_from_gov"
        elif is_top_secret:
            res_key = "top_secret"
        elif is_secret:
            res_key = "secret"
        elif is_restricted:
            res_key = "restricted"
        elif is_public:
            if is_private and not is_gov:
                res_key = "nongov_private"
            else:
                res_key = "public_gov"
        elif is_private and not is_gov:
            res_key = "nongov_private"
        elif is_gov:
            # Default gov without specific tag is treated with restricted caution
            res_key = "restricted"
        else:
            # General fallback to private/other
            res_key = "nongov_private"

        rule = cls.CLASSIFICATIONS[res_key]
        return {
            "key": res_key,
            "rule": rule,
            "detected_is_gov": is_gov,
            "detected_is_private": is_private
        }
