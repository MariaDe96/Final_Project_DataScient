# 🧠 Predicción de la Depresión con BRFSS 2022

## 🎯 Objetivo del proyecto
El objetivo de este proyecto es contribuir a la **prevención temprana de la depresión** mediante la creación de un algoritmo de clasificación que permita predecirla a partir de diversos factores, tales como:

- **Factores demográficos** (edad, género, estado civil, nivel educativo),
- **Salud física y hábitos de vida** (actividad física, calidad del sueño, hábitos de consumo),
- **Experiencias traumáticas o conductas de riesgo**.

Esta aproximación busca apoyar la detección temprana de personas en riesgo y facilitar la implementación de estrategias preventivas en el ámbito de la salud pública.

---

## 📊 Origen de los datos
Los datos utilizados en este proyecto provienen del **Behavioral Risk Factor Surveillance System (BRFSS)**, que es el sistema de encuestas de salud por teléfono más importante de EE. UU. El BRFSS recopila datos a nivel estatal sobre residentes, abordando aspectos como:

- Comportamientos de riesgo para la salud,
- Condiciones de enfermedades crónicas,
- Uso de servicios de prevención.

Se seleccionó la **data de 2022** por ser la versión más reciente y completa al momento de iniciar el proyecto.

---

## 📋 Sobre el dataset
El dataset utilizado refleja casi **medio millón de personas encuestadas** y contiene **328 variables** (features), abarcando una amplia variedad de aspectos relacionados con la persona y la entrevista. Sin embargo:

- Muchas variables no eran relevantes para nuestro análisis (e.g., detalles sobre la fecha de la encuesta, la forma en que se realizó, la ubicación geográfica, etc.).
- Gran cantidad de registros presentaban **valores nulos** o códigos para representar la **no respuesta voluntaria**, lo que requirió un tratamiento específico para su correcta interpretación o eliminación.
- Existen variables que son **cálculos derivados** proporcionados por el propio BRFSS (e.g., índice de masa corporal), calculadas a partir de otras variables base.

Esta situación requirió una cuidadosa **fase de preparación y limpieza de datos** para garantizar que solo la información relevante y de calidad estuviera presente en el análisis final.

---

## 🔍 Tratamiento y selección de features
Se realizó un **proceso de selección en dos etapas** para reducir el dataset inicial de 328 variables a un conjunto final de 39 **features**:

1. **Primera selección**: Se preseleccionaron 79 variables con base en la relevancia de la pregunta y la variedad de aspectos cubiertos (socioeconómicos, demográficos, hábitos de vida, estado de salud, etc.).

2. **Segunda selección**:
    - Se descartaron variables con un **alto porcentaje de nulos** o cuyos datos estuvieran representados por códigos de no respuesta.
    - Se eliminaron variables **redundantes**, priorizando aquellas calculadas por el BRFSS que resumían de manera clara conceptos básicos (e.g., índice de masa corporal o puntajes para representar diferentes tipos de **consumo de nicotina**).

De esta manera, alcanzamos un conjunto final de **39 features**.

> ⚠️ Es importante destacar que este análisis quedó limitado a personas de **más de 45 años**, dado que para rangos de edad más jóvenes la cantidad de registros válidos y consistentes era insuficiente. Se recomienda para futuras etapas profundizar en el **tratamiento de datos nulos**, de manera que pueda ampliarse el análisis a otras franjas de edad.

---

## 🔄 Recodificación y tratamiento de datos
Los datos del BRFSS venían **codificados de origen**, asignando diferentes códigos numéricos a cada categoría de respuesta. Sin embargo:

- No seguían un **patrón claro y estándar**, especialmente para variables ordinales, por lo que fue necesario un **proceso de recodificación manual** para garantizar que:
    - Las respuestas **binarias** estuvieran en formato estándar (`0 = No` / `1 = Sí`).
    - Las respuestas **ordinales** estuvieran correctamente alineadas para reflejar la progresión lógica de menor a mayor intensidad, frecuencia o importancia.
- Se realizó un análisis de **outliers** para detectar y **excluir registros atípicos**, garantizando que no afectan al análisis ni al modelado.

Contando con un volumen de datos lo suficientemente amplio, esta depuración no comprometió la robustez de los modelos, pero sí contribuyó a obtener un dataset más claro, coherente y significativo para representar la realidad analizada.

---

## 📊 Análisis gráfico
Se llevó a cabo un análisis gráfico para evaluar la incidencia de cada variable en relación a la variable objetivo (**ADDEPEV3**), con el fin de obtener una primera aproximación a los factores clave para la predicción de la depresión.

Por ejemplo, en la imagen a continuación se observa que, en el caso de la variable de género, las **mujeres** (codificadas como 1) presentan un porcentaje mayor de personas con depresión en comparación con los hombres:

![Incidencia de la depresión por género](data/image_1.png)

También es importante destacar que la variable objetivo (**ADDEPEV3**) presenta un marcado desbalance: alrededor del **80% de los registros corresponden a casos negativos** y un **20% a casos positivos**, lo cual es típico en este tipo de estudios médicos y de salud mental.

Esta situación de desbalance guió posteriormente la selección de técnicas para el modelado, especialmente en la evaluación de métricas para valorar la capacidad del modelo para detectar correctamente los casos positivos.

---

## 🤖 Modelado y evaluación de modelos
Se probaron diferentes enfoques para obtener un modelo que generalizara bien y atendiera correctamente al desbalance presente en la variable objetivo. Se evaluaron tanto modelos lineales (**Regresión Logística**, **KNN**) como modelos no lineales (**Random Forest**, **XGBoost**, **CatBoost**) e incluso **redes neuronales** simples.

El objetivo principal fue maximizar **Recall** y **Precisión**, sin perder de vista un **Accuracy** aceptable, considerando que en este tipo de problema médico la prioridad es **no fallar en la detección de casos positivos**, aunque ello implique aceptar una mayor cantidad de falsos positivos.

Se decidió mantener la distribución desbalanceada original, para reflejar de manera más fiel la realidad de este fenómeno en la población.

Tras la fase de pruebas y ajuste de hiperparámetros, el modelo que mostró el mejor balance global fue **CatBoost** con umbral optimizado, alcanzando los siguientes resultados:

### ✅ Resultado Final del Modelo

| Métrica     | Test  | Train |
|--------------|------|------|
| **Accuracy**  | 0.78 | 0.79 |
| **F1 Score**  | 0.57 | 0.59 |
| **Precision** | 0.48 | 0.50 |
| **Recall**    | 0.70 | 0.72 |

Estos resultados indican que el modelo presenta una buena capacidad para detectar personas en riesgo de depresión, alcanzando un **Recall** alrededor del 70%, lo que respalda su utilidad para apoyar en estrategias de prevención e intervención tempranas.

---

## 🌐 Despliegue en Streamlit
A continuación se muestran algunas capturas de la **interfaz final en Streamlit**, donde el usuario puede responder a las preguntas del modelo y obtener una **predicción en tiempo real** junto a una **interpretación clara** del resultado.
https://final-project-datascient.onrender.com/

<div style="display: flex; justify-content: space-around;">
  <img src="data/image_2.png" alt="Página principal" width="48%" />
  <img src="data/image_3.png" alt="Resultado de predicción" width="48%" />
</div>

---

## ✅ Conclusiones
- El análisis de datos del BRFSS permitió construir un modelo de clasificación robusto para detectar personas en riesgo de depresión, alcanzando un **Recall del 70%** en test, indicador clave para este tipo de problema médico donde la prioridad es no dejar casos sin diagnosticar.
- El cuidadoso preprocesado de datos, que incluyó selección de variables, recodificación manual y análisis de outliers, resultó esencial para obtener un dataset claro y significativo.
- El modelo final (CatBoost) alcanzó un balance adecuado entre **Precisión**, **Recall** y **Accuracy**, demostrando su utilidad para apoyar estrategias de prevención e intervención tempranas en contextos de salud pública.
- El hecho de mantener la distribución desbalanceada original garantiza que los resultados reflejen de manera realista la ocurrencia del fenómeno en la población.
- A futuro, sería valioso profundizar en el **tratamiento de datos nulos** para poder ampliarse el análisis a otras franjas de edad, especialmente para evaluar la incidencia en personas menores de 45 años.
- El despliegue en Streamlit proporciona una **herramienta interactiva y explicativa** que podría integrarse en entornos sanitarios para apoyar a profesionales en la toma de decisiones y facilitar la comunicación de resultados a los usuarios.

---

## ⚡️ Limitaciones del Proyecto
- El análisis está limitado a personas de **más de 45 años**, dado que para otros rangos de edad la cantidad de registros válidos y consistentes era insuficiente para garantizar conclusiones fiables.
- El modelo refleja la distribución original del BRFSS, que presenta un marcado desbalance (80% casos negativos vs. 20% positivos), lo que limita la capacidad de detectar todas las posibles señales de riesgo.
- El BRFSS depende de datos autoinformados, lo que puede introducir sesgos y datos imprecisos, especialmente en variables subjetivas (e.g., hábitos de sueño, calidad de vida).

---

## 📚 Referencias
- [Behavioral Risk Factor Surveillance System (BRFSS) - CDC](https://www.cdc.gov/brfss/index.html): Página oficial de BRFSS.
- [CatBoost Documentation](https://catboost.ai/docs/concepts/python-reference_catboostclassifier.html): Documentación oficial de CatBoost para detalles de modelado e implementación.

---

## ⚖️ Licencia
Este proyecto está licenciado bajo la [Licencia MIT](https://opensource.org/licenses/MIT), lo que significa que:

- Puedes usar, copiar, modificar y distribuir este proyecto para cualquier propósito (incluido comercial).
- Solo debes incluir una copia de esta licencia junto con la mención al autor original.
- El proyecto se ofrece "tal cual", **sin ninguna garantía** de funcionamiento, responsabilidad o idoneidad para un propósito específico.
