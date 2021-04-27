hocomoco <- "/Users/minjunp/Documents/baylor/mpra_project/liver_enhancer/HOCOMOCO/hocomoco_entropy.txt"
learned_filter <- "/Users/minjunp/Documents/baylor/mpra_project/liver_enhancer/HOCOMOCO/learned_filters_entropy.txt"
hocomoco <- read.delim(hocomoco)
learned_filter <- read.delim(learned_filter)

t.test(hoco, learned_filter, alternative="less", var.equal=FALSE)
