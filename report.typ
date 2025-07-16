#import "base.typ": *

#let report(
  title,
  subtitle,
  toc: false,
  justify: true,
  lang: "es",
  size: 11pt,
  title-size: 20pt,
  margin: 1.2in,
  number: "1/1",
  doc
) = {
  assert(("es", "en").contains(lang))
  // The default font is Libertinus Serif. I could have used Libertinus Math for
  // math, but IMO the default New Computer Modern feels more "professional".
  set text(size: size)
  set text(lang: "es", region: "AR") if lang == "es"
  show raw: set text(
    font: "Source Code Pro", weight: "medium", size: 0.9 * size
  )
  set par(justify: justify)
  set page(numbering: number, number-align: right, margin: margin)
  align(center, {
    text(title-size, weight: "bold", title)
    if (subtitle != none) {
      v(1em, weak: true)
      smallcaps(text(calc.max(0.9 * size, 0.6 * title-size), subtitle))
    }
  })
  if toc { outline() }
  doc
}
