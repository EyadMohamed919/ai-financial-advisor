Router: trained on 1605, tested on 403 held-out messages. Accuracy **87.3%**

| Route | Correct / total | Accuracy |
|---|---|---|
| transaction (تسجيل معاملة) | 104 / 112 | 92.9% |
| spending_query (سؤال عن مصاريفي) | 50 / 56 | 89.3% |
| forecast (توقع) | 26 / 31 | 83.9% |
| market (أسواق واقتصاد) | 41 / 45 | 91.1% |
| advice (نصيحة) | 36 / 45 | 80.0% |
| manage (تعديل/مسح) | 41 / 42 | 97.6% |
| greeting (تحية) | 30 / 32 | 93.8% |
| out_of_scope (خارج النطاق) | 24 / 40 | 60.0% |

Most common confusions (gold → predicted): out_of_scope → advice (7), out_of_scope → spending_query (5), spending_query → manage (4), transaction → spending_query (3), advice → spending_query (2), advice → out_of_scope (2), advice → market (2), advice → forecast (2), forecast → advice (2), forecast → transaction (2)

Sample errors:

- `el mafrood a2sat el doyoon matzedsh 3an kam % men el moratab?`: gold advice, predicted spending_query
- `اشرحلي طريقة كرة التلج في سداد الديون المتراكمة`: gold advice, predicted out_of_scope
- `المرتب بيتبخر في أول أسبوعين من الشهر ومش فاهم بيروح فين`: gold advice, predicted spending_query
- `Is it better to invest monthly savings in gold bars, money market funds, or bank certificates?`: gold advice, predicted market
- `ashtery 3arabeya mosta3mala cash wala zero b loan?`: gold advice, predicted out_of_scope
- `How can I protect my long-term savings from inflation in Egypt?`: gold advice, predicted market
- `How can I stretch my remaining 2,000 EGP to last until payday in 10 days?`: gold advice, predicted transaction
- `How can I budget effectively when my freelance income fluctuates every month?`: gold advice, predicted forecast
- `ناوي أبطل أطلب دليفري من طلبات الشهر الجاي عشان أوفر`: gold advice, predicted forecast
- `متوسط صرفي اليومي الأسبوع الجاي المفروض ميزيدش عن كام عشان مكسرش الميزانية؟`: gold forecast, predicted advice
- `emta hawصل لهدف ال 100k law ba7awesh b nafs el tari2a?`: gold forecast, predicted advice
- `تفتكر مصاريف الشوبينج الشهر ده هتعدي الشهر اللي فات بقد ايه؟`: gold forecast, predicted spending_query
- `When will all my ValU installments be fully paid off?`: gold forecast, predicted transaction
- `امتى الرصيد بتاعي هينزل تحت الألف جنيه لو مبطلتش طلبات أونلاين؟`: gold forecast, predicted transaction
- `التطبيق ده جامد جدا`: gold greeting, predicted out_of_scope
- `سلام بقى عشان داخل اجتماع في الشغل`: gold greeting, predicted out_of_scope
- `kansel order talabat abo 280 geneh`: gold manage, predicted transaction
- `تصنيف مصر الائتماني الأخير عند وكالات التصنيف اتغير ولا ثابت؟`: gold market, predicted out_of_scope
- `When is the next Central Bank of Egypt monetary policy committee meeting?`: gold market, predicted out_of_scope
- `faydet shahadat el bank el ahly el gedida kam %?`: gold market, predicted spending_query
- `3a2ed ozoon el khazana 12 shahr kam ba3d el dareeba?`: gold market, predicted spending_query
- `kam calorie fe sandwich el shawarma?`: gold out_of_scope, predicted spending_query
- `ازاي انضف بقعة قهوة من على قميص ابيض قطن؟`: gold out_of_scope, predicted advice
- `كام جرام بروتين في ١٠٠ جرام صدور فراخ مشوية؟`: gold out_of_scope, predicted market
- `عايز اغير العملة الأساسية للجنيه المصري`: gold out_of_scope, predicted advice
