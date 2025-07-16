// Descargar los fonts para distintos íconos de:
//  https://github.com/oblador/react-native-vector-icons/tree/master/packages
#let icons = (
  ok: (0xf164, (:)),
  bad: (0xf165, (lower: 1/7)),
  pin: (0xf295, (family: "octicon")),
  warn: (0xf21f, (family: "octicon")),
  idea: (0xf0eb, (:)),
  up: (0xf062, (:)),
  down: (0xf063, (:)),
)
#let icon(c, family: "awesome", lower: 0, size: 8pt) = {
  if type(c) == str {
    let (c, args) = icons.at(c)
    icon(c, ..args)
  } else {
    let fonts = (
      awesome: ("Font Awesome 6 Free", "Font Awesome 6 Free Solid"),
      awesome-solid: ("Font Awesome 6 Free Solid", "Font Awesome 6 Free"),
      octicon: "Octicons",
      material: "MaterialIcons",
      feather: "Feather",
      ionic: "Ionicons"
    ).at(family)
    text(str.from-unicode(c), font: fonts, size: size, baseline: size * lower)
  }
}

#let todo(msg) = box(stroke: fuchsia, inset: (x: 1pt), outset: (y: 2pt), msg)
// No funciona en ecuaciones: https://github.com/typst/typst/issues/4698
#let sfrac(c) = text(fractions: true, c)
#let loc(c) = text(style: "oblique", [#c#h(-1pt)])
#let vs = loc("vs.")
#let ie = loc("i.e.")
#let eg = loc("e.g.")
#let viz = loc("viz.")
#let iid = [_i.i.d._]

#let sgn = math.op("sgn")
#let var = math.op("Var")
#let cov = math.op("Cov")
#let cor = math.op("Cor")
#let sd = math.op("SD")
#let bias = math.op("Bias")
#let mse = math.op("MSE")
#let med = math.op("med", limits: true)
#let mode = math.op("mode")
#let argmin = math.op("arg min", limits: true)
#let argmax = math.op("arg max", limits: true)
// https://github.com/typst/typst/issues/4802
#let bar(c) = math.accent(c, "\u{0305}")
#let avg = bar
#let distas = sym.tilde
