#import "base.typ": *

// https://forum.typst.app/t/4608
#let config = state("slides.config")

#let slides(
  size: 10pt,
  lang: "es",
  accent: luma(245),
  libertinus-math: true,
  doc
) = {
  // Beamer uses 128mm but with 4:3 and 1cm margin by default
  set page(width: 128mm, height: 128mm * 9 / 16, margin: (x: 0.8cm, y: 0.6cm))
  // Emojis are quite broken, use #base.icon for now.
  // https://github.com/typst/typst/issues/4755
  // https://github.com/Myriad-Dreamin/tinymist/issues/1821
  set text(font: ("Libertinus Sans", "Noto Color Emoji"), size: size)
  set text(lang: "es", region: "AR") if lang == "es"
  show math.equation: set text(font: "Libertinus Math") if libertinus-math
  show raw: set text(
    font: "Source Code Pro", weight: "medium", size: 0.8 * size
  )
  show raw.where(block: false): it => box(
    fill: accent, outset: (y: 2pt), inset: (x: 1pt), radius: 1pt, it
  )
  show raw.where(block: true): it => align(center, block(
    fill: accent, inset: (x: 10pt, y: 7pt), radius: 2pt, it
  ))
  config.update((accent: accent))
  doc
}

#let title(title, author: none, email: none, place: none, size: 18pt) = {
  align(center, (
    context box(
      fill: config.get().accent, inset: size * 0.8, radius: 0.5em,
      // https://www.reddit.com/r/fonts/comments/fbxw1m
      text(size, font: "Jost", title)
    ),
    v(1.5em),
    text(12pt, author),
    linebreak(),
    underline(text(9pt, email)),
    v(1em),
    smallcaps(text(10pt, place))
  ).sum())

}

#let section(size: 20pt, title) = {
  pagebreak(weak: true)
  align(center + horizon, text(size, heading(level: 1, title)))
}

#let slide(
  body-align: horizon,
  col-align: (left, center),
  col-width: 60%,
  density: "normal",
  title,
  ..bodies
) = {
  let subtle(size, str) = text(size, font: "Jost", fill: luma(130), str)
  set page(
    header: context {
      let headings = query(heading.where(level: 1).before(here()))
      if headings.len() > 0 { align(left, subtle(7pt, headings.last().body)) }
    },
    footer: context {
      align(right, subtle(5pt, counter(page).display("1/1", both: true)))
    }
  )
  let bottom = if body-align == top { 10pt } else { 0pt }
  block(
    below: 0pt, inset: (top: 3pt, bottom: bottom),
    text(16pt, heading(level: 2, title))
  )
  // https://github.com/typst/typst/pull/4390
  // Space between lines and tight list items
  let leading = (compact: .55em, normal: .65em, cozy: .75em).at(density)
  // Space between paragraphs and non-tight list items
  let spacing = (compact: .87em, normal: 1em, cozy: 1.25em).at(density)
  set par(leading: leading, spacing: spacing)
  align(body-align, {
    let bodies = bodies.pos()
    if (bodies.len() > 1) {
      let columns = if col-width == none { bodies.len() }
        else if type(col-width) == ratio { (col-width, 100% - col-width) }
        else { col-width }
      grid(columns: columns, align: col-align, gutter: 1em, ..bodies)
    } else { bodies.first() }
  })
}
