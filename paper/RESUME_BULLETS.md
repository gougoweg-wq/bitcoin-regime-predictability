# Строки для резюме

Все цифры взяты из работы и воспроизводятся кодом. Подставь своё имя и ссылку на репозиторий.

## English

**Project title:** *Is Bitcoin's Direction Predictable? A Pre-Registered Out-of-Sample Test of Regime-Based Forecasting*

**One-line version**

- Independent quantitative research: built and stress-tested a regime-based Bitcoin forecasting model in Python; validated on 22 months of unseen data under pre-registered criteria (7 of 12 hypotheses confirmed).

**Detailed bullets (pick three or four)**

- Designed a sequential Bayesian model over six market regimes; identified one regime with a statistically significant drift (t = 3.48 in development) that replicated out of sample (t = 2.72).
- Built a volatility-regime forecast with 60–66% accuracy, replicated at 58–65% on a 22-month hold-out, with block-bootstrap confidence intervals excluding chance.
- Audited my own earlier result of 60.76% accuracy and showed it to be a backtest artefact (true value 57.14%, p = 0.059, Deflated Sharpe Ratio 0.05, Probability of Backtest Overfitting 0.68).
- Evaluated 16 model improvements, including gradient boosting with meta-labelling and neural networks, under walk-forward validation with embargo; documented all negative results.
- Backtested a trend-regime exposure rule that returned +19.8% on unseen data while the asset fell 17.1%, cutting maximum drawdown from −53% to −25.5%.
- Assembled a survivorship-free dataset of 913 Binance perpetual futures (prices and funding rates, 2020–2026) and tested cross-sectional carry and momentum strategies.
- Calibrated the research procedure by Monte Carlo: on 6,000 simulated series with no predictability, the discovery step alone gave a ~20% false-positive rate, cut to ~2% by requiring hold-out replication; a result as strong as the reported one arose twice.
- Quantified tail risk (99% Expected Shortfall 10.0% vs 13.5%; Student-t fits with ν ≈ 2.2–2.6) and ran power analyses showing why directional claims on a few hundred observations are unverifiable.
- Packaged the model as a tested Python library with a CLI and CI; an automated test rewrites future prices and asserts that no past forecast changes, and regression tests recompute the paper's headline numbers on every push.
- Applied Newey–West inference, multiple-testing corrections, and a single-use hold-out with frozen code; released a fully reproducible repository.

**Skills line**

Python (pandas, NumPy, SciPy, scikit-learn, LightGBM), time-series econometrics, Bayesian inference, backtesting and overfitting diagnostics, reproducible research with Git.

## Русский

**Название проекта:** «Предсказуемо ли направление биткоина? Проверка режимной модели на независимых данных с заранее объявленными критериями»

**В одну строку**

- Самостоятельное количественное исследование: построил и проверил на прочность режимную модель прогноза биткоина на Python; на 22 месяцах независимых данных подтвердились 7 из 12 заранее объявленных гипотез.

**Подробно (выбери три-четыре)**

- Разработал последовательную байесовскую модель по шести режимам рынка; нашёл режим со статистически значимым дрейфом (t = 3.48), который подтвердился на независимых данных (t = 2.72).
- Построил прогноз режима волатильности с точностью 60–66%; на независимых данных точность составила 58–65%, доверительные интервалы исключают случайность.
- Провёл аудит собственного раннего результата (точность 60.76%) и показал, что это артефакт бэктеста: реальное значение 57.14%, p = 0.059, Deflated Sharpe Ratio 0.05, вероятность переобучения 0.68.
- Проверил 16 усилений модели, включая градиентный бустинг с мета-разметкой и нейросети, методом walk-forward с зазором; задокументировал все отрицательные результаты.
- Протестировал правило управления позицией по тренду: +19.8% на независимых данных при падении актива на 17.1%, максимальная просадка −25.5% против −53%.
- Собрал свободный от ошибки выжившего набор данных по 913 бессрочным фьючерсам Binance (цены и фандинг, 2020–2026) и проверил на нём стратегии керри и импульса.
- Откалибровал саму процедуру исследования методом Монте-Карло: на 6 000 искусственных рядах без предсказуемости этап поиска давал около 20% ложных находок, требование подтверждения на отложенной выборке снижало их примерно до 2%; результат такой силы, как в работе, возник дважды.
- Оценил хвостовой риск (Expected Shortfall 99%: 10.0% против 13.5%; t-распределение с ν ≈ 2.2–2.6) и провёл анализ мощности, показавший, почему заявления о точности направления на нескольких сотнях наблюдений непроверяемы.
- Оформил модель как библиотеку на Python с командной строкой и автотестами: один тест переписывает будущие цены и проверяет, что ни один прошлый прогноз не изменился, другие пересчитывают ключевые цифры статьи при каждом изменении кода.
- Применил оценки Newey–West, поправки на множественные сравнения и одноразовую отложенную выборку с замороженным кодом; выложил полностью воспроизводимый репозиторий.

**Навыки**

Python (pandas, NumPy, SciPy, scikit-learn, LightGBM), эконометрика временных рядов, байесовский вывод, бэктестинг и диагностика переобучения, воспроизводимые исследования в Git.

## Как подавать на собеседовании

Сильная сторона работы в том, что ты сам опроверг свой первый результат. Формулировка: «Я получил 60%, не поверил, перепроверил и нашёл три ошибки. Потом построил процедуру, которая такие ошибки ловит». Для позиций в количественных финансах и анализе данных это ценнее, чем высокая цифра точности.

Не пиши в резюме «модель предсказывает биткоин с точностью 60%+»: это не то, что показали данные, и на собеседовании это быстро выяснится.
