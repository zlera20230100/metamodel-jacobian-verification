# Figure style guide

The manuscript uses one colour vocabulary across schematics, curves, markers,
and bars. Colour is never the only identifier; line style, marker shape, or
hatching is retained wherever series must remain distinguishable in grayscale.

| Role | Colour | Hex |
|---|---|---|
| primary analysis, surrogate signal, or main curve | blue | `#3775BA` |
| second quantitative series | medium blue | `#6B9AC4` |
| third quantitative series or second physics family | light blue | `#A9C5DF` |
| pale comparison bar or uncertainty support | pale blue | `#DCE8F1` |
| rejected, stressed, or adverse contrast | muted warm accent | `#C76B3C` |
| reference, bound, or neutral context | dark grey | `#4D4D4D` |
| complete-FD or background support | light grey | `#CFCECE` |

The figures use one blue family, neutral grey, and one warm accent. Green,
yellow, saturated red, and unrelated categorical palettes are not used. In bar
charts, series are also separated by the same hatch
sequence (`///`, `...`, `\\`) and thin black edges.

All plotting scripts use Times New Roman for Latin text, STIX-flavoured math,
editable Type 42 text in PDF, and vector PDF as the manuscript source. PNG
files are high-resolution previews rather than the typeset source.

## Framework and flow diagrams

- Use square-cornered, Visio-style engineering boxes arranged on aligned rows
  and columns. Connectors are continuous orthogonal polylines with consistent
  line weight; diagonal or freehand routes are not permitted.
- A connector terminates at the node outline. Its shaft and arrowhead must not
  enter the node interior, cross a node, or cover node text.
- Connector labels sit in reserved whitespace beside the route. Do not place a
  label on a line and do not use a white label mask that visually breaks the
  connector.
- Titles, annotations, legends, and callouts require visible clearance from
  every box, connector, and arrowhead. Text must remain wholly inside its node
  unless it is an explicitly external annotation.
- Source scripts must run a geometry check for connector/node intersections.
  The final PDF is also inspected at its actual manuscript size before release.

## Manuscript integration and pagination

- Figure QA is not complete at the standalone-file stage. Recompile the full
  manuscript and inspect consecutive page spreads after every change to a
  figure, caption, algorithm, float option, or section length.
- Do not release a build with an avoidable large blank region, a figure-only
  float page, or strongly unbalanced page density. Normal page margins and
  unavoidable end-of-section space are not defects; float-induced empty areas
  are defects and must be reflowed.
- Preserve reading order: first citation, relevant figure or algorithm, and
  interpretation must remain adjacent. A deferred float must not allow later
  explanatory paragraphs to overtake it.
- Treat neighbouring figures and algorithms as one layout unit. Check the
  complete page sequence for float order, caption proximity, legibility at
  manuscript size, and balanced use of the text area.
- A warning-free compilation does not pass this gate by itself. The upload PDF
  must receive a fresh all-page visual inspection before its checksum is frozen.
