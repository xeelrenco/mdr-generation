# Punti da chiarire con Renco

Checklist di decisioni aperte sul generatore MDR. Aggiornare quando Renco risponde.

---

## Scheduling (date e predecessori)

### 1. Politica di fallback quando resta un ciclo

Il fallback alfabetico resta **solo** per cicli sconosciuti.

**Comportamento residuo** (dopo il sort topologico sui nodi aciclici):

1. I nodi coinvolti in un ciclo non corretto vengono aggiunti in coda in **ordine alfabetico** per `TitleKey`.
2. Per ogni documento, entrano nel calcolo solo i predecessori **già processati** (`finish_by_key`).
3. **Nessun documento viene escluso** dal calcolo date.

**Riferimento codice:** `mdr_generator/12_schedule.py` — `_topological_order`.

---

## Decisioni già prese (non in attesa)

| Argomento              | Decisione                          |
|------------------------|------------------------------------|
| Separatore titolo MDR  | Solo `\|` per suffissi 3b e 3d     |
| Lingua suffissi titoli | Inglese (prompt 3b/3d)             |
| Colonne debug schedule | `schedule.debug_columns` in settings |
| Ciclo List ↔ Summary   | Summary prima; List dipende da Summary. Catalogo `DocumentPredecessors` già senza arco inverso (verificato 2026-09-21). |
| Durata documenti       | `raci_matrix.DocumentDurations.Days` da Excel `Durata più conservativa`. Non più mediana timeline. |
| MANHOURS               | `raci_matrix.DocumentDurations.ManHours` (valore di catalogo, non più Days × 8 in pipeline). |

---

*Ultimo aggiornamento: 2026-09-21*
