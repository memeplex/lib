#import "base.typ": *

#let report(
  title,
  subtitle: none,
  author: none,
  email: none,
  toc: false,
  justify: true,
  lang: "es",
  size: 11pt,
  title-size: 20pt,
  margin: 1.2in,
  page-number: "1/1",
  heading-number: none,
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
  set page(numbering: page-number, number-align: right, margin: margin)
  set heading(
    numbering: if (heading-number == true) {"1."} else {heading-number}
  ) if (heading-number != none)
  show heading: set block(below: 0.8em)
  show link: underline
  align(center, {
    text(title-size, weight: "bold", title)
    if (subtitle != none) {
      v(1em, weak: true)
      smallcaps(text(calc.max(size, 0.6 * title-size), subtitle))
    }
    if (author != none) {
      v(0.7em, weak: true)
      let author-size = calc.max(size, 0.5 * title-size)
      text(author-size, author)
      if (email != none) {
        " " +  underline(text(author-size, "(" + email + ")"))
      }
    }
  })
  if int(toc) > 0 { outline(depth: int(toc)) }
  doc
}
