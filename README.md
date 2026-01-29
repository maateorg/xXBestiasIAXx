# ⚡ EcoMind: Agentes de Auditoría Energética Inteligente

> **Eficiencia energética en tiempo real impulsada por la velocidad de Gemini 2.5 Flash.**

![Estado](https://img.shields.io/badge/Estado-Hackathon_MVP-success)
![Gemini](https://img.shields.io/badge/AI-Gemini_2.5_Flash-4285F4)
![Python](https://img.shields.io/badge/Python-3.11-blue)
![TotalEnergies](https://img.shields.io/badge/Sponsor-Cátedra_TotalEnergies-red)

## 💡 Sobre el Proyecto

Este proyecto fue desarrollado durante la **2ª edición del Hackathon en Agentes Inteligentes y Grandes Modelos de Lenguaje**, celebrado en la **Escuela de Ingeniería Informática** y patrocinado por la **Cátedra TotalEnergies de Analítica de Datos e Inteligencia Artificial**.

### El Problema 📉
El análisis de informes de consumo energético y normativas de sostenibilidad es un proceso lento y manual. Las empresas pierden oportunidades de ahorro por no poder procesar grandes volúmenes de datos no estructurados en tiempo real.

### La Solución 🚀
**EcoMind** es un sistema multi-agente que ingiere documentos técnicos, facturas y logs de consumo para detectar anomalías y sugerir optimizaciones automáticamente. Gracias a **Gemini 2.5 Flash**, logramos una latencia ultra-baja, permitiendo análisis conversacional instantáneo sobre grandes conjuntos de datos.

---

## 🏗️ Arquitectura de Agentes

Utilizamos una arquitectura orquestada donde cada agente tiene una responsabilidad específica, comunicándose entre sí para generar el informe final.

```mermaid
graph TD
    User[Usuario] --> Manager[🕵️ Agente Orquestador]
    Manager --> Reader[📄 Agente Lector de Datos]
    Manager --> Analyst["🧠 Agente Analista (Gemini 2.5)"]
    Manager --> Auditor[✅ Agente de Cumplimiento]
    Analyst --> Report[📝 Generador de Informes]
