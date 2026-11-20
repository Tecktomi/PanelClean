# Datos de terceros incluidos

Esta carpeta contiene material que **no fue creado por el equipo de PanelClean**. Se incluye en el repositorio para que, al clonarlo o descargarlo, no haya que buscar archivos adicionales. Su licencia es distinta de la del resto del proyecto.

## `open-mantra/`: OpenMantra dataset

| | |
|---|---|
| Origen | <https://github.com/mantra-inc/open-mantra-dataset> (rama `main`), publicado en GitHub por `mantra-inc` |
| Artículo | R. Hinami, S. Ishiwatari, K. Yasuda y Y. Matsui, «Towards Fully Automated Manga Translation», AAAI 2021. [arXiv:2012.14271](https://arxiv.org/abs/2012.14271) |
| Licencia | Creative Commons Atribución-NoComercial 4.0 Internacional (CC BY-NC 4.0). Texto completo en [`open-mantra/LICENSE.md`](open-mantra/LICENSE.md) |
| Obras | Cinco series de manga japonés. El repositorio original no nombra a los autores de cada obra; los derechos de las imágenes pertenecen a sus titulares originales |
| Cambios | Ninguno. Los archivos se copiaron tal como vienen en el ZIP de la rama `main`; solo se cambió la carpeta donde están (de `open-mantra-dataset-main/` a `datasets/open-mantra/`) |

### Contenido

- `images/<obra>/ja/*.jpg`: 214 páginas en japonés.
- `annotation.json`: por cada página, los rectángulos (`x`, `y`, `w`, `h`) de 1592 bloques de texto, con su texto en japonés, inglés y chino, y los rectángulos de 1069 viñetas (`frame`).
- `README.md` y `LICENSE.md`: documentos originales del dataset, sin modificar.

| Obra (carpeta) | Páginas | Bloques de texto |
|---|---:|---:|
| `tojime_no_siora` | 46 | 334 |
| `balloon_dream` | 38 | 314 |
| `tencho_isoro` | 40 | 311 |
| `boureisougi` | 36 | 274 |
| `rasetugari` | 54 | 359 |

### Uso en PanelClean

La evaluación (`evaluacion/`) lo usa para medir la detección de texto y el tiempo por página, como fuente de las frases japonesas de los pares sintéticos y para etiquetar a mano el fondo de regiones reales. Más adelante puede servir para entrenar un detector (tarea T1.2.3). Para medir sin sobreajustar, las páginas de una misma obra deben quedar todas en el mismo conjunto (entrenamiento, validación o prueba); la obra `tencho_isoro` está reservada como conjunto de prueba.

El uso es académico (curso TEL-328, UTFSM). La licencia no permite usar este material con fines comerciales.

### Cita

Si se usa este dataset, la licencia exige atribución y sus autores piden citar:

```bibtex
@article{hinami_aaai_2021,
    author={Ryota Hinami and Shonosuke Ishiwatari and Kazuhiko Yasuda and Yusuke Matsui},
    title={Towards Fully Automated Manga Translation},
    booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
    year={2021}
}
```
