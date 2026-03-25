convert m1_full.jpg \
  -filter Lanczos \
  -distort Perspective \
    '926,670 0,0  1174,668 248,0  1174,712 248,42  923,716 0,42' \
  -crop 252x41+0+0 \
  +repage \
  m1.jpg

convert m2_full.jpg \
  -filter Lanczos \
  -distort Perspective \
    '767,587 0,0  1003,575 248,0  1005,616 248,42  766,629 0,42' \
  -crop 252x41+0+0 \
  +repage \
  m2.jpg

convert m3_full.jpg \
  -filter Lanczos \
  -distort Perspective \
    '736,640 0,0  967,636 248,0  970,675 248,42  735,682 0,42' \
  -crop 252x41+0+0 \
  +repage \
  m3.jpg
