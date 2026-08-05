# LotteryAI Evaluation Dashboard

- 生成日時: `2026-08-05T00:07:11.428857+00:00`
- Full Run: **ok**
- Dashboard schema: `1.1`

## 全体

- Optimizerのゲーム内ランダム差が最も高いゲーム: **LOTO7**
- 事後評価が最も少ないゲーム: ミニロト, Numbers3, Numbers4

## LOTO6

- 次回抽せん回: **2126**
- 採用Config: `experience_loto6_1`
- 採用元: `experience`
- 事後評価: **参考値** (9回)
- 平均最高一致: 1.5556
- 平均1口一致: 0.9556
- 最大一致: 3
- Optimizer selection_score: 2.453168
- Optimizer Random uplift: 0.251867
- Experience履歴: 3件

### 次回予想

1. 6・8・21・26・35・42<br>2. 2・10・18・27・37・38<br>3. 1・5・15・28・36・40<br>4. 7・12・16・19・23・43<br>5. 4・11・14・20・33・34

### 注意

- 事後評価が9回のため、成績は参考値です。

## LOTO7

- 次回抽せん回: **689**
- 採用Config: `evolution_01_no_delay_strict_experience_loto7_1`
- 採用元: `evolution`
- 事後評価: **データ不足** (4回)
- 平均最高一致: 2.25
- 平均1口一致: 1.25
- 最大一致: 3
- Optimizer selection_score: 3.656509
- Optimizer Random uplift: 0.3185
- Experience履歴: 3件

### 次回予想

1. 9・12・15・22・29・34・35<br>2. 1・4・13・17・26・31・36<br>3. 3・7・8・18・20・23・32<br>4. 6・11・14・19・21・27・28<br>5. 2・5・15・22・24・30・33

### 注意

- 事後評価が4回のため、長期成績は判断できません。

## ミニロト

- 次回抽せん回: **1399**
- 採用Config: `evolution_02_experience_miniloto_1_experience_miniloto_2`
- 採用元: `evolution`
- 事後評価: **データ不足** (2回)
- 平均最高一致: 1.5
- 平均1口一致: 0.8
- 最大一致: 2
- Optimizer selection_score: 2.287626
- Optimizer Random uplift: 0.1926
- Experience履歴: 3件

### 次回予想

1. 3・11・14・20・31<br>2. 4・16・19・21・27<br>3. 2・10・22・23・29<br>4. 7・13・18・28・30<br>5. 1・12・17・24・31

### 注意

- 事後評価が2回のため、長期成績は判断できません。

## Numbers3

- 次回抽せん回: **7042**
- 採用Config: `experience_numbers3_1`
- 採用元: `experience`
- 事後評価: **データ不足** (2回)
- 平均最高一致: 1.0
- 平均1口一致: 0.45
- 最大一致: 1
- Optimizer selection_score: 3.026392
- Optimizer Random uplift: 未評価
- Experience履歴: 3件

### 次回予想

1. 473<br>2. 239<br>3. 067<br>4. 671<br>5. 328<br>6. 932<br>7. 526<br>8. 760<br>9. 147<br>10. 852

### Numbersバックテスト

- 平均最高位置一致: 1.305556
- 1口平均位置一致: 0.316667
- 平均最高順不同一致: 1.85
- Straight率: 1.11%
- Box率: 7.78%

### 注意

- 事後評価が2回のため、長期成績は判断できません。

## Numbers4

- 次回抽せん回: **7042**
- 採用Config: `local_01`
- 採用元: `local`
- 事後評価: **データ不足** (2回)
- 平均最高一致: 2.0
- 平均1口一致: 0.5
- 最大一致: 2
- Optimizer selection_score: 2.131251
- Optimizer Random uplift: 未評価
- Experience履歴: 2件

### 次回予想

1. 7461<br>2. 1647<br>3. 8514<br>4. 4158<br>5. 9225<br>6. 5229<br>7. 7830<br>8. 0387<br>9. 8703<br>10. 6932

### Numbersバックテスト

- 平均最高位置一致: 1.533333
- 1口平均位置一致: 0.387778
- 平均最高順不同一致: 2.205556
- Straight率: 0.00%
- Box率: 0.56%

### 注意

- 事後評価が2回のため、長期成績は判断できません。
