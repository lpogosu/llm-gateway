# llm-gateway

Единая OpenAI-совместимая точка входа к нескольким LLM-провайдерам: маршрутизация с
fallback, семантический кеш, лимиты по ключу, учёт токенов и денег, метрики и трейсы.

[![CI](https://github.com/lpogosu/llm-gateway/actions/workflows/ci.yml/badge.svg)](https://github.com/lpogosu/llm-gateway/actions/workflows/ci.yml)
![Python 3.11](https://img.shields.io/badge/python-3.11-blue)
![License MIT](https://img.shields.io/badge/license-MIT-green)

---

## Задача

Как только в компании появляется больше одного сервиса, который ходит в LLM, каждый из
них тащит свой клиент к провайдеру. Дальше всё предсказуемо:
