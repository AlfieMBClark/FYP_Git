"use strict";

// Inlined training land-penalty overlay (base64 PNG) — self-contained so the
// Penalty button works whenever app.js is deployed, no separate asset needed.
const LAND_PENALTY_PNG = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAA+gAAAKACAYAAAAPc1JnAABDjklEQVR4nO3d6Y3cOqIG0LLRqRjOyKl0DE6lMzI6mffjjp7VshaSIsXtHGAw13YtqiqJ5Cdu314AANChz5/vH8t///jz+1fI40L9+PP71+fP94/l/1OPESDGt9oHAAAAsZbQfBbM1487s32N7XPW/y6sAyUJ6AAADOksTF+F8pDnLM/Tyw7k8r32AQAAAAB60AEAGExoz/ndXu+UXniAMwI6AADdWoaYpz4313EsxyCkA3cI6AAADOlsIbkSQVpIB+4yBx0AAAAaoAcdAIDhPN17DpCDgA4AwPD2Arst0oDWCOgAAAxnHbqttg70QkAHAGAoZyu7C+dAyywSBwAAAA0Q0AEAGIoh7UCvBHQAAJqQK0gvr/P58/1j77+Phr8D1GYOOgAAj7sK49vV1UNCdUzAX15P7zrQkrfaBwAAwFxCQ3FMT/e6dzz09YVzoDWGuAMAAEADDHEHAOAxIb3WZ1ukXfWq6xUHemaIOwAAjzqa/30WvgVvYAZ60AEA6Na2V12QB3pmDjoAAF1ab6G2/N1ZL7zt1YDWCegAAADQAAEdAIDuxe6ZDtAic9ABAOjS0Xxzi80BvdKDDgBA0z5/vn9sg/VZ0N57PEAP9KADANC89Wrte/uh7wVyPelAb/SgAwAAQAP0oAMA0Kylp3uvN/yqF/zHn9+/znrW9aIDrRHQAQBo0tHK7DHB+iikA7RIQAcAoBvCNjAyc9ABAOhCbM95yWMBKEFABwAAgAYY4g4AQPNCe88tAAf07K32AQAAQKyjIex7e6Tnei+hHyhNDzoAAM07Csepq7uHsAI88DQBHQCApp3tZb5+TK6ebj3mQC0WiQMAAIAG6EEHAKAZS6/10dD1kj3ny+vrOQdqEdABAGjKErqPFnwLXQguZd904RyoSUAHAKB5V4vErXve9/776Hlnrw3wNAEdAIBm7A1xX//9npjeb8EcaJlF4gAAAKABb7UPAACAud3pyV73iB/1pMf0xgPUZIg7AADZ7YXk7eJuOeeHny0mJ5ADvRDQAQDIImSeeMhj18+JmVd+dFPg6vkArRDQAQC47Swch/aax4ZpARwYjUXiAAAAoAF60AEAeITeboBzetABACZXIjh//nz/WF53/fp7i7kB8B8BHQBgUEcheatkaN6+r150gGOGuAMADGi7pdn23/b+PldQ3+sxT1kALuV5AD0T0AEABpcScvdWXj/6uxwhOmYbtpg91QF6Yog7AAAANEAPOgDAQHL2JOupBniWgA4A0LklRIeG5/X87qvnpb5uyOMB+EpABwAYRMpK7SHPSQ3cueanA8xCQAcA4J9F33KE8+1rA3DOInEAAADQAD3oAAAcipnfnjKMHoC/BHQAAKLs7Yue+hoA/CWgAwBwaR2oz4J57IryAPwloAMA8I/YXnKBHOA+AR0AgCRCOUBeVnEHAACABuhBBwBg19m8c73nAPkJ6AAAfHEUvpeQLpwDlPFW+wAAAKhjG7TvbJsGwH3moAMADOyot3vv7z9/vn9sh7UvoT2m91zQB0hjiDsAwARCe8vPHhc6tH29D7o90QHC6UEHAACABuhBBwCYxF5P9lUPecy/nz0OgGsCOgBAxz5/vn/EzPnOMYR9+1yBHCAPAR0AoHNH871j9i4PnZMOQDkCOgDAQJYe9dRgve0V3/vz3fcAYJ9F4gAAAKABetABADp21It9txd93Ut+NvzdPHSAfAR0AIAO7A0rfzIc31lMDoAwAjoAQMOOtjY72/KsRIDWUw5QnoAOANChlD3LUwnnAM8Q0AEAOlBzaLtgDvAMq7gDAABAA/SgAwA0poV9xmu/P8CMBHQAgIYJyQDzENABABojlAPMSUAHAGiEYA4wNwEdAKAywRyA18sq7gAAANAEAR0AoLJlX3MA5maIOwDAwwxpB2CPHnQAgActe5zXPg4A2qMHHQDgIXrOATijBx0AgEt6/QHK04MOAFDAtrd8HXD1pAOwRw86AAAANEAPOgBAJnrGAbhDQAcASLQO5HtztHsO7D/+/P61t+J8z58JoHUCOgDADUchvecgG7MgXM+fE6A1AjoAQGY9htbYVdp7/IwArRPQAQAyWoJr673poYG8xWMHGJVV3AEAAKABetABAAo4m5u+LMBW47ju9JzXPG6AGQjoAACFtBZmQ8L50TEvz23tMwGM5K32AQAA9Gqvl7zlANvb8QLMRg86AECAHsJt7J7lZ0PW7X8O8DwBHQDgxMjBNHZrtbWRvxeAWqziDgAAAA3Qgw4ATGdvr/Kzx43m7HOHfjfrxwKQh4AOAFRzFvDuDL++Y4bQuTeHPuX7nuG7AniSgA4AVBW6SNmTtsE1Noi2uojc6/X12EJGELT8WQBGI6ADAFWEBL4lHLfcm350bMtxtxJsY77DVo4ZYDYCOgBQ3Hpec0z4qxHM74by/EcUL3Ruva3UANoioAMA2aQGvJrD2be2n6GHHnILugGMwTZrAAAA0AA96ABAk0IXMVv+O7QX/mwbsZj57rV7pHs5TgDCCegAQFUhW60dDTsPHY6e6/jWr18r+ArmAON6q30AAMBcYoPjWQgvscp7i8HWHHOAca3LeD3oAEBWNVcIvxPUQ4/vagX3XAvH3e0pt385QLuOyngBHQAo7umQWHrP7xI3H1JvLoSOIBDUAfK5cxP0rMwW0AGArGoGwbNGz3Yu+d3jvBPSS24rJ4gDlBe7JklouW+bNQAAAGiAReIAgCxq9NxezQdvTcme89er3c8NMLKr3vSYst8QdwDgttLBsPSc8lSlA3cKIR3gOaHTnULrCwEdAPgitXERGwzvrrgeOt98/X4lw2uLYX0htAPE2y4Et61HYkZxCegAQJAl7OZa5ObsdXKF2LO90UOPpZTYGwd7zz1r3IU2/IRygHtiFh7d+/eUOk9AB4BJpQa43D3FIaE1Zo/y2sG0xv7vAORVa1SUVdwBYEJ3QuPd58Y+P2XofK2GVcvD3AFon4AOAAAADTDEHQAmdacnPLan+Klh3mdz0WsuEmeYO0Bfao2IEtABYGLbRWxCFzGLef3t82MXQAsV8zlyCXnPlhawAyCMgA4ANGO9pczRv4cGzyeCc46AD8CcWlo/REAHAIoTnAF4UuwOIKWPJ5SADgDcsu5NPxq+LqADUFrKaKqWwvnrJaADAJFi560vDEMHIKeQeuXqMQI6ANCFkN7w2N5xIR2A0pa6aRu+z9ZIaYV90AEAAKABetABgNfrdX+Ls9TFePSmA5DqqLf89TrfcaRVAjoAEL2gztU2bNvXrrFHOQDj6S1wx3qrfQAAQH2pDZ6reerbOeejN6wAyG+m3UD0oAPAgPYC8dKTHbtIzlUPeOq2NrM0tgC4Z6abuwI6AAzqqBd7T8gQ9KuQLnADUMJMAd0q7gDQsc+f7x/bYLz3d2dSGj4WegMgtx9/fv86Wuxt7+9GrHsEdAAAAGiAgA4AnVr3HFz1JBwt5hazuvqIPRUA1He0oOjRCK+RFx41Bx0AOhUbmM8aMuvXCplPvjfEPWWxOAD6kXu9kTs7iIwYzl8vAR0AmhKykFtK4+hoznjICu4pr7v+97NV44V5gLYt5Xjtm7B7u5LUOI7SBHQAqOROI+cq5IZsixZyLDENsqseeiu9A/TtiZA+avAOJaADwMOeGB6YI3DvvUfIFmxXjwWgb0/ecJ0tsAvoAPCQEo2Zp4aLp4b00scFQB3roe+lyvnZwvnrJaADwON6DKwhPe9357MD0JfUcH7npu/obLMGAAAADRDQAeBhPfYKhKz6vuzFfrUn+9XrADCubfm/9+eZ6whD3AGggNAhe3tbxbQ6FNz2aACs7U1/iqnTZg7iRwR0ACjgamuzqwZLywFYUAdgsQ7pMfWXcL5PQAeAE3s93Cmv8Xo9s3/sosbq7qXfC4A2bUP6+t9Cpkjxl4AOAAdKNiruBNncoTjn9mgCO8Cc1tuu1T6WngnoAHAgJqCHzCXPua1MavCN2Qrtbi+8sA4wl7ORYoJ7GAEdADZSes5Dwmfuxknswjt3e+1z9PoL6QDjOqsrBPQwAjoAU0sdxr5+Xs3GyNFwwhaDcOuL3wFwz145L5jHsQ86AAAANEAPOgDT2ZtnHTvffPv8UkLn8emZBqCW7QJxT9aToxHQAZjK3pZnubZR24rd+/zO+xo+DkArhPJ0AjoA0znqLd+G9eXPR/PpQgNxyONTVr49GwkAADUJ6WkEdACmcNZTnrqFWO4wHPqad7c/A4CctvWXcJ7urfYBAEBJpQJszWC8HgFwNQpg/XgAKEUoz0NAB2BoOYZ+ny1+U9N2CP4S2Fs5PgDGJpTnZ4g7AFM4m3e+/HfqXugtEdABeIqAnp+ADsBw7q7MHhNwW+tVB4A9Z/Vi6G4klPe99gEAAAAAetABmMTVKu4pPeFXPQtH27QBQCvO6ik96M8T0AEYXs6AfLVC+jroW00dgJaFbD8qpD9LQAegSTHzyPeCb0o4vvN+oa8tpANQ016P+VVQF9KfI6AD0KyrkH4Vdu82KI6C/52AfvS6AFDS2Y3i7ZadTx8bfwnoADQjdiu00KAbE/JLBWk96ADUtp2GdXfXE/IT0AFoxjqghwxbXz/nTOxwdyEdgBEJ4+0T0AHoUmwwjw3FpcK0kA5AC4T1NgnoADQldpuzq4Vu7gThkmH6atj90c0F27YBkIuQ3p7vtQ8AAAAA0IMOQEV7i8LFLvy2t4Bcrh7mkiuv3+kJ14sOQC560dsioAPQlJSV2fdWoW19cTdz0QFomeBeh4AOwKOOtkw7esyekG3Tcvd+l+qdF9ABqCl0RJrA/gwBHYAsjoLmVYWeWvlfbcNWcmG3nCFdQAegBQJ4GwR0ALIosR/58pzYld1jXj9WjpCu9xyAVgjmbbGKOwBZ/Pjz+1eJSn5vfnnIc5bn5T6mu6F6ueEgnAMAWwI6ANlchc6rPcvPnpcStNfz0XMG9Turr6//f+/fAOAp6p72GOIOQHFX88VDHh/iaDh86mJ0ADAyAb09AjoAjzrrRd8uGBc791zgBoAwwnmbDHEHAACABgjoAFR11Out9xwAmI0h7gAkORqOnrKV2h3COQDEM8S9TQI6ALfErMwuSANAXYJ52wR0AJIt259dBW+93ABQn3DePnPQAUi2Dd6x25lpKADAM9S5fXirfQAA9Oto3vniarj70gNf6rgAAHpiiDsAQc5Cb2jP+Z2h7st7tBK+9z5zK8cGAFt60PsgoANwKEfgXDcIcr9ejL33TllxPuRGQWs3EwBAQO+DgA5AlNjQWSOsnm37triaF38W6AVvAHojoPfBInEAAADQAD3oAAQLXY29dg/z3lz3lN7v1j4XANylJ71tAjoAh7ZbqK33PW91kbSzhegMUQdgZsJ5+wR0AE7d2Qrt6SAsgAPAMQG9fQI6AEmOtlATkgGgXUJ62wR0AKIJ3wDQhtjdSkKfTx0COgCHtnPQ9/4eAGjLdjTb1eg2Ib0dAjoASYR0AGhXaM+6cN6Wt9oHAAAAwD2hQftstxPq04MOwKmQVdy3Q+j2/g0AKONoSDv9EdABSHIV3M+CufnsAJCPQD4OAR2AL1L2PT9aTO7scQBAHmfzzYX3vnyvfQAAAACAHnQACtBTDgDP0EM+FgEdgGCxi88I6gDwDEF9DAI6ALeEzj/fPj50objtyrTL3wn/x3w/AHMS0vsnoANQzFVIPAqSy98LmnF8XwDzEs7HIKADkEwYrG9vhAEAcxPW+yWgA5Bkb+sWIfFZwjkAwvhYBHQAkgmGz1mHccEcgIWAPhYBHYAgZ3PFj55z1GgYNVg+MQdcOAdgj6A+BgEdgGDbUHgUFmdcrOzJcP56CegA7BPU+yagA/B6vfbnlN99vVyv1brS4VwwByCWoN6nt9oHAEAbYivyWYPiNozPOFoAgLYJ5/36XvsAAOjL58/3j9y97T368ef3rxLhfO97Xb7zo38HgNfrb91U+zhIJ6ADAABAA8xBByDZjEO7a6+iHvv+huADzEUPet/MQQfgi+0+23v/vQhpBAiH13KHfovKAcxHMB+DHnQAgiwBfT0X+ir8lQyKMXuwlwi+e7aLx129d2rv9tE89djXAWAsQnr/BHQADoWEvqMg/NQCameO3v9Oj/Xe5405rtjv9Oy9hHIA1o7qD8G9HwI6ALtiw9+28r8bHp9sTMTM5777vO1zzj7nXo987HsDMA9BvH8COgCHYoawnz0/ptc3Zl57aE/4Xm/+3UZMTEDeC9d3h6lb/A2ALQG9fwI6AF+cDQvPuRd3yGulBO877hxL6Gtvw7mQDUBOQnrfBHSAyV3NG1/+7qjCTx0envo6TyndQ117uzYAxiOc9882awB8sVe53w3ny2M1HP7yfQBQQs7RbjxPDzoAp4uRHT0+Zw/zrPOp9aIDkJtg3jcBHYBD5kuXJaADkJNw3r/vtQ8AgHZtw/ny55hh8Bz7/Pn+YSgiADmoR8agBx2AXfbgrkOvOgApBPQx6EEHAACABuhBB+CQXtzn6UEH4A496X0T0AG4tJ0nvf5zjiA56yruAFCKoN4nQ9wBuHRWycc2APYeL5wDQF7q1j7pQacp672V3fWDvmgIAEB7tKn7IqDTjKPGvUIF2nO2wvveY82rBoBnaUP3SUCnCcI5tO0qWIdcq8I5ADxPe7ovAjpNENChvu1CcEf/vna0uJt90wGgLdrVfRDQaUJIAx+o62xYe0oPu9AOAM/Qru6HgM5j7vS+AfWFjHQJCd0xjzd3HQDSaE/3yTZrAAQJGfr+48/vX8v/rl7vbLeG5TU+f75/xCxIBwD8Z6k/3eTuix50HqOnHPpyFKBDK/qjETKh7j4fANDe7o2ATnEWgIM+XS0at35MbsI5ANSjnV6PIe4kWw893Q5DXT/m6jXWz1//r8xRA3dtr8/QIe1ntq/hxh4A1KMtXo8edJLtNdKvHnP2+G1vXUjvHVDP3eHruV4LAMgntO0d084nnIBOtNiL0cULfQpZnC1kFfar13lqTntLRvosAIwntd6/ej7XBHSCLItFpYRtQ1Whb0ejWbaLyIVe66nBdIRQO8JnAGAOsduoHj2XOOagAwAAQAPeah8A7bOHIsxtvR/5+o74Xm/w0d32vcfNVqboTQCgJ3dGvN19z5nrTEPcObSdNxoy13T73JkvLhjR2XD3vceflR3bBSGv9Bjqcyy0AwC9is0CIevfjE5AJ8leY/Ko8TzrxQUjSgnoZ/++PCb0BmBLQTakzGvpeAGghtSQPmuGENBJEjNc9WphKaA/V9dx7AicmMfVELJAnmAOAPti2v6z96IL6NwSs7q7YA7jOLu7fVax9hBcQ2887D2+h88HALWEZIEZAvrZGj4WiSNJTCN03ZAX0mEs62t6G9qXheVyvl+poe6xDQYAIN4M4Xtrr32095glK+lBJ8nexRWz0NP2dWa5QGEUZ9NcSgxVz/2asb3kV68jvANAnNipcr3nhdC2goDO/0sJyzFD3HO8H1BfyIJuucJ66QB8NUQ/5PnCOQA8Y4TccLVOj4BOsJiV24+McFEBf5Wek10qAN8N5uvXENABoIyjaXQ9Cu28MAedL2IbminD2oG+hV73LQfXvbnzqa+T76gAgLWzabWt5ouzEcbrYz5qh3wvf4igEQsjOasQ9yqe5b9jKtJcle7R+94N5602CgCgZ2fthXUwb6UeXrcj1ovjhkwHPHqcIe78I2WYZ+ge6MAY7qw7ESLnEPQjLc+NBwCOtZYxcrYH9KDzj+1dqeXPVxdCjgY10IeYu9exd7y3c9hTtms7u0mQa/s35RsAlLU3Mq/3zHHVFtKDTpCQHvKUHjWgf3tbrsU8/ug5uYef56q8rdwOAOXFzDmvvb/60crsKW0Gi8QRZBvEj+Z0np2AtS8coIzYrRmX55xVuEeh/+pG4N7cr/Udd8EaAPrQ+mJwZ+4csx70wdQ+kc96xvbuLPV4wQFfbVdEL31dn43oCemdTwn/Ie8DAJSzV//u1fEt5ouYdoOATnZnPeWtXjRAPtvtRXJe8yGrooYMuT/bAgUA6E+ptkcuoW0YAZ0sjuZZtHhxAM95OqSf9eTnCONCPQC0r7cMsm63COhkFbtYFDCms7nmVzfxzm74he4run0tAGAuveYQ26xRTK8XBXBf6Lzu9aJu2+3Pts8VtgGA0QnoZKPxDPPaBuur0TRne5keuXqs3nMAYNFrO0BABwAAgAaYg042FoeDucWuQRFzZ/votc52huj1zjkAkE9vueSt9gEAMIbYQB46T/3sdXurdAEAzuhBJwurtwMh9sL4uryIKUuOVnsHAFjrKZsI6NwmnANHliHoIQH67HEWgAMAUuxt+doyAZ2szuaDAuM6qvjWZUJoSD97ve37wUy2o01cHwDpWs0sVnEnq1ZPdOAZ2/3Mcw8/39vCDWawvZZibnwB8F+5eTatrhUCOgC3rSu8dXBY9/LF3sC7M6zdzUJGpMccIN22XbL++1rHtMcQdwCyOavkQnrTY4L12Xz10PdprVKGI3vXhh51gHit38S3zRoAj4hZMO7sNdZ/jln4Zduzn3oM0IK9c9h5DRCnxfWz9KADkE1ICD/q6QutIPd6DUMfJ8DQqx7mTQLUktoB0Fo4f70EdAAK2S4Ut/7/lNfZvtbZ46AXKbschD4egHMCOgBTOOodP9uObfvYvddsfWEXSBF6XgvoAGn2boa2GM5fLwEdgMzuhPO9fw99HxjBWQhfX0u5tzAEGEXqiL1WCOgAZBVzd/oqzG/thRe9iowm5Jy2rgLAV0fT33oL6vZBB6CYo/BwtBfp2XPWz9v7u7tB5cef3796q8QZk9ANEC6kM2DbTmi5nBXQAcjqamX1bQXZUihuucJmLs5FgDBHa9RsR/StH9NS22NLQAegmJChuilz0IUXRtdy4xGgFXuj37aBvbfy9K32AQAwnt5WW7fgFrWcnXvOSYBjR8F7+fvegvnCInHsOlptGSBWaMhIDcm5w7VF53ja0U4He+df7LoNAKPoaau0OwR0AIq6Cg93AvbdlaythE0r7jQynb/AyI7q6hHD+esloANQ0BPB+U7INrSdlqQ0Nu2JDozqaETbqMF8YQ46AMXEBod12I6tgK9Wj495PNQQO71MOAdGFFs/t7YjzF0COgBN2AaNo9ARG8TPXkOwoUUhPUWjNUiBsYXUu2dl2tJGSAnvV6/dGkPcASiqdAiOWdRNbyM9umpYOqeBloXeFF+H8KsblSFhfR3Oewrq9kFvmAoX6F1KL3js4z5/vn+Evo9yldE4p4HWLfX0tk5e9jDf28t8e/M9ZoG4dRjvcT90AT0zFSVAPrFl6nrv014qYrjDeQ607qwH/Sx4H91kX/9/SHjvbUqQIe4ZrH/03k4AgFL2Ksy94WZnYveDPjqGo/cz5J1ehPYWOZ+BVsWu/3JVpuUaCt8aAf2GmZb7B+aWWsEdhfSQILFXkd8ta0tv2wYlxdyUAuhFar07agepVdwzSll84Og5o51oQN9yhPOz17wanpZSrsY8f13JCzi0SJsA4KttjuppIbgzetALiAnXMQseAPSk9N7j6574lO1bBHFaF7vysHMa6M3RsPSU547SwSmgP2CvcjVvHRhZyKrq26HuucrB0D2kc7wXPOHuCBKAHoSE9Rkyk4D+gL3enRlOLmBeVwH9iQp3G9QFF3qwd67GXh+x1x9AS1LKvJGylW3WIsXMTzxa9n+kEwigpr1ydnsjVDinF7naB9oaQK9y3ZDsmR70SGfb9JwNy1BRArMp2Use89ojVt6M52iNhJxTP0LWYXBDC2jBzNlJQN8ImTe5fdyd1Y1nPvkAzsQOWbsqV4UOrjy9kODRGgxPthHO3h+glpkzkoD+P6n77uV6z5lPQoAjoTdNQ54DV+6sJpz6XqUDeuxok6t56nrYYS6lrvmQRWJn7dA0Bz3RnRN11pMNIFXInFp7mHPX0dox6//P+V7L625fO+dibkfXztln3f739nnaLzC+s7Ipx+uelW/r8unJG6et0IP+P6k/9NH+e0I4QD4hoWD0CpvnHTUMc/UoPd1GOOohTxnarycdxra3xlauvHTGCGM96Mn27iotJ5QKCyCvkEq65MrVVsVmEdOjvj1v1s+tcT7tvWfqvHttHZjDnWs9tqy7e+NwFHrQ/yf0R9dAAyjvqJJOGV57Vb5bVI4QKedJj20G5z3weu3PEY8pH8563nssG5/0VvsAehPT0DNPC+C+mAbBevjx1RDcnPN8GZu6HJhNrrrRkPV4etBXcs2zWNTaMgWgd7G93qF7PC/PFcoJMVud7boA9tztQT+iZ32fHvSVvcUQjh63/vPVyarCA0gTs6DV9jHbIL78WZlMiBkbivZCB16veyPMQlZoP3o//jN1D3rOEyRlr14A9q1HHe3dYQ/pYc89KooxmZq2z/UCc4vd3ix2MThl7TEB/cSdE8eJB3BPziHq5puzpn4+nnrnGoG5XY0o3lvrJXQUMmGmDuiLnHeDYp8DwFelAoKQPjd1cxjXB8wttKyMCebCexwB/X9ChqifBXArFALkISBQkjr6mmsQ5pJSLm6noilb8/le+wBaEXJSLYsOrf9uu+CQkxMA2mWhwDDaMzCHs2v9qKy0+npZetA3jhYmCjl5nZwA9wlPlKa+DuNahDFZHLNttlnbWJ+gTlaA5wgDlLC+4a5eD+d6hPG5ztukBz0Td58A7tNYIMXeAoDq5HtcizCWozW0lJXt0YOeiZMbAOoQJoGZySFj0YMewV0mgPKELXJQX+fjmoQ+KPfGMH0Pesy8NCc9ALRLPQ086Wrr5Sfe2xob45m+B/3OReRCAChHrx0h1MXluRbh2NVOTynbmN19X/r2be/E6P0Hv/pMexdL7AXS+3cE0APBgC2rstfhWoR9R1szh14zyjG2dgP6Yn2CPX3yxFa+T1ccLiaAZwgGrKl/63AdwnOUc3M7Deh7QnueY0+sdSBvvRJw0QA8p/U6geeph5/j+oPnKePmFh3QFyHDN7bDymOe2yoXDEBdPdYd5KEOrsM1B3Uo8+aUFNBDe7l76A2P4SIBaMdI9QvH1L1tcL1BXcrCeXyPefCPP79/zRrOAWiLxsrYljZH7ePgP34LqEuumkdwD7rA/ZWKCqAt6qgxqF/b5RqDNHem9yoT55M8B31mLhSAtqx3G1n+W/3WH/Vru1xPkNdRPaUcJGqIOwC0aN2gWf5bIwfqMD0Brm0X0F5fN+t/c3NsPgJ6JBUOQD+U2f3wW7VrGxBa/K1aPCYItRfC11tQP39E8/r8+f6x/l+NYzDEPYILBKBf6ru2qFPHcXcr3Rzb7/a8hS/sUUbWU3vqwdtTb9QzFwhA/87Kco36Z6lXx7EXzo9s14kocTzWn6BX63N377+Vm8+p/V3rQb9Q+wcC4DnqxPLUq+M4CuexQ+JzXHd60BmVMnM+AvqKCwCA2HpRj1049ez4jsL53WHwMJNtD3rt4+FZ0w9xd9IDsLZXLwgScQzJnE/KnM0cN7eEfUaytyOJoD6fqXvQnegAxDgbujtzfbpQr85re0Pm6npw7cAxZencpt1mzYkPQKy9/Z1H790I+Vz2vZ7X3vkfErhjFpdbP855xgzctJrbdD3oCnYAchu1Lg0ZPqxe5fVKuwZih6db74HZbG98KW/nMM0cdCc0ALmFDOPtNVCE9pw/cSy066nz27nGbLbnvGtgHsMPcTccCoASYobx7tVDrdZNe/Xm3p9bPX764PyBv1wPrA07xN2JDkApuYbkttjDrv4k1dkNqaPhua2d/1CT7dV4vQYM6E5mAJ5yVIeeLSTX2rB49Sa1jNYGhRyUyXQf0J3EAPRk6UncW8W6dJ2szuRJV3ujx57vLY44gZyWc1xZPbeuF4lz8gLQm+0Qxs+f7x+xW05th0DG7DkNJT1xk0lIZ2TKa7rtQXfyAjAC8w0ZzbZtmWPu+Z2ed2iVOefs6aoH3YkLwGjUbRBOOKdHRyM/tsPZhfW6YkezldJFQHeSAgD05aj9dqf3HHq0DuLb899Np/qObp5s/+6psqj5Ie4KZQCAccS0PQ1tZzQh228+eTyzqxnEj3TRgw4AwFye2t0AWmIV9zhna15c3QgJWS+jhu+1DwAAgPFtdyw4I5wzkh9/fv9a/rcXwFsJhj2JKU+ObL/3VsqbKkPc9wrd1H3/3GUCAGhb6laArTSYIcXVKu1yTLqjsiH25l6LK+k3Mwf9KKCfrabnpAYA6MNVgzr08dCbdWg8O9/lmnOlyoTWvvdmAnqK1r5MAACOHXW87M0F7bmNCnv2OiQF82Mly4CWv/Nu56C3/KUCAPCv9Vzc5e9aXagJcluf6+th1W5G/Wvm76S7VdwV2gAA/Zu5Ac7clnN/6T2Xb57Ry+KTXQV0Jy8AQL9abxgD9eVYnb3nveabCugxC8Tl0PMPBwAwgqPVrWscCzyttRXEa8t17ff8fVZbJK7Wl5b6ea+ON3Yp/5TPf7VNw9H7xb4PANCO2RrwwjmzmeXaviKc/+fxgH409n9vKELql2vRhWO9n7AAMItZgrm2GrMa/dqOkbMc6P17fSygx4bl2C9W4R6u95MWAEZWeopfq7TlmM1M1/cZ4fyr4gE9drU8wfw5I5zAADAK241p1zGXdU5yvefT+3dZfJG41C9eAQ0AjMxitV9p+zG6vUA+Wzh3nV+rtkjcmtU765mpQACA0gzVvEf7j9Gtp/3OdI0/eW33/r1+r30Ae1urKZwBgN7kbr9oD8F4XNdceSSgH93FWP+9YF6H7xwA7itVn6qnYVyzXN9PLkree+/56/XQEPe9XvLS70m4EU5kAKihRjtqZNqIzGiGa/zJkP7E+5TUxBx06hrhRAaApz3dhhq9vtYmZUajX9cLc9DDCeiT6/0EBoCn1Wg7zVBf732vsdv1Qo9muL5fr3rXcW/fb/VF4qhjlDkaAMAYtu0S7RRGpz3+jN5u8OlBn4xCAADS1G4zzVyH1/7uIYfc13CPe6iXvJZ7+y6OCOiTGOWEBYAaareX1OP/qf07wF25ruXttdBLGVHqGu7l84cwxH0CI52wAPC0FkJhiWPYe80WPiuMKkebvOetqUsc94jTBN5qHwBljXbCAsCTUhqUP/78/tVqA3p7XMufl/aCdgPklfOaOltIsWWtloetEtAH1sMFCwCtaq1RmTrftLXPkWqUz8E8SrfFtfX/vck4AnPQBzXSSQoAT2q5bXS3fu913urr1fbvAmdKzTvP+dpPKDXEPfdr1qYHHQDgf1oNgbkaoT03ZtfH3urvBFslw/mMei7DQgnoA5rhxAWAXFpv+KrXoU/C+Vd31ueYqRy0ijsAMK2YxuLTDcQRVyfOxfdCD9bly52QfXS+93YdpC662dvnvMsc9EHNdiIDML5lkbTURYF6avOox6/19Hsyt9xrR+R4zdrOrt/eP9tdAvqgZj+xARhHaFul5e3NYqnHw43ymzO+1Ou658Udj1xdtyN8xlQC+oBmPqEBaN9R22Opv0q1TbYBPjXQny1Wtq2D774+17Rl6cndkT8jlA9H5fAIny0HAX0wTmwAWtZTuyNnnRozCiDXe86kp/OKucVc4yMObd8joH9lFfeBOKkBaFnpEBXTI353qOnZ8/ceY4swmNdSNt0J56O285WH/9KDPohRL1oA+perrZHauC2xQFOsvWOYpQH+JO1aWpNyXY/ec341zWl2etAH4GQGoEV3w9Kd+i0kEIc8N1fgu+pVB8YjnJNCQO+cCxaAFrU25zr2eEr1xGp8AzPTe35NQAcAssoZbmvseW6YdL9G2mqP/t2ddz4a4TyMgA4AZBPTwCzVKBPO57YdBSG0U4tw/lXKYnkzEtA75uQGoCW5wvndxdMEMl4v7STGMNp5PNrnKUFABwBuSw3EZ8/TkCOX0msLQCznIkdss9YZjRUAWhPSlojpMQ99XuwxtRrS1O11tHYeMJaj67q1BTRpjx70TrhIAejV0+H87L3Up+TS6g0f6ltPs4ndtlEZhR70hrlAAehJ6Aq9qT1IsWG+pzaOOv95qedH7+caZek5567vtQ8AABhXSjj/8ef3r22v09Xzet9fXMB7Xuj5sX5cT+cU0Cc96A1ZD5VSAQDQo5gV2GNDdezw0FbaOOr0tl2dJ6Hts1bON+q6M8JCWcHrJaA3ywUKQM9CbzbH3JQODVIpz83pbIir+r1Nd0dgaE+zyDWlh3kJ6I1xcQIwktTgE9s+KfGaNd5TO6Ce9W8V+ztoT/N66T0nD6u4AwBRYnqC91a6zjlUuEajtsbNAMoTkIh1dc4I56QQ0AGAR5VcBfuJxeJq9MTTLjdj2OO8IJUh7g1RQQPQujvDgGNeO0TIgnCt71Wt7h9Lq+cZ+eVc80I5wJpt1hqw3U4GAHoQsv1ZjNR5v2fHELOq/NP2tpIT8Np39Bv57YglA7DHEPeKXJAAjGCZUx7ai52z/gsJRevjayVEnfXsW+29P62cV9TnXOAuAb0ClS4APYrpqQ79t6e0cAwxzo43dNi+9kYZvte55fr9nUccEdAf5EIEoGdnveSEydEWCP0N9MTntTcCxPUwF9cTTxDQM3HBAkCYWUJNri2Y7igxrWB2s5y/fJXzGnI9ckZAj3R0QV3NuwMA5tJSkNObvi+k/dbS7wiMzzZrgbaVWuj3pjIEYDTaDv3RHtnnXCZE6g4TOV6L+ehBP7BdgMV+hgDwn9b3FodQesq5ok3P0/SgP8CFDcCotCPapx0SzvnMVsr1Y+ordwjolbhAAejRtt2gN71t2hvxnMssUq+fvXPItUgoQ9wrsaoqACMQZspIXfvm6PmEM+yd3FyPxPhe+wBm5mIFgL7l3npp+d+d99G+2Cdsj6vEOX/nNV2D3KEHvQIXLQCMIVfoO2sbxLyHNsa5oykaIY+lXa39VuvjcU0Syxz0TK7m4Lk4ARiBdkN+IW2Eq6lx9jkP4/wlxN1ryfXIHQL6TS4+AGai3ZBHSM9tTBtDIAjnHG5D6AKTP/78/hX6m1m7gREI6IlcwADMRpshH+2IepzHee0tqhd7fofcYNq+9vKcmJtT69eIme4AT/onoMfcpZqRixeAWWkf5KM9UZ/zOY+nzuWco0SEc1pmFfdAR6uqAgDQH+26PJ660ZF7x4QSrws57A5x14v+LxcvAPxLeyGdtkWakDnLOV+PMC2ez6lD7qGmoYa45z72HBfzdpsFi7gAMKJe2w61aROEu3uO2VKtPOcz3Hca0HsL6zHblKQ+P1TqapMA0KOe2gst0Q44V+K8OtuqLvd7jSJmdXTnNNyzG9Bfr+MhISMXXnsFSmyPd67vR+EGQG9GbiOUor7/q/T5c7cjZzbOTajjMKAvjgLqjAVYjTuuCkcAejJj++COEttRjSL3uRS7FdeMZjm3oGXftnOkX6+4xTfWexCWO0xeL4UmAH3QJgg3Wt1+NQKz1gJuFo47Nto5CL37tldgzlQojUQBC0ALtCPCjFZvH+0MlPs1Uwnp+0Y7D6F3318v4RwAiPP58/1jr82gHfHXjz+/fx2Fn5aCa0u255XwWJbvF9rzbe8vRy30Z6GwBaCklOlxo0rtlc0V0Fuq84/OgZTFdkt1HqV8XyOf2y2dP8B/3rZ/MXIhNIuZFpABgJ7k7Dlvoa4PaTf23i4Z9SZUz78JjEwP+sAUvACUcrUPsrZEmXq4hYAe89vemffdUi967mNogXYitOn7+g8tDpci3WgVCQB9UP+U10NbLWZbs701DUqcR3e+tx6+81AjfRYYzf8PcVeZAgDkUSoA1Q5WMe3F7WPvhmNtVWAGhrhPonaFDsA4tBOu9VTvhg6bT/ndY3rRr55Ta0/0UsdRU0/nJ8zmn4A+QqHDPoUxADmM2la4qidDP3cv9e1Tv2PM95ornJ/1uOf6fXq+Dno5R2FGuz3oi54LHs4pmAFIMWPbYJY6s4XftmTvecjvmLLifAvfW6pZzm3oyWlAf736LnQ4t9xdVjgDEGrmdsHI9WWrv2uOldy3v9vZosix7aJWv7cYI5/X0KPLgL41QkHEV+thYAppAI5oA/xn1LpyhN83dpqC+ej/GfWchh5FB/TXa5zCiGMKagC29ur/pb6YqW1QOtTVrIN7/R1Dh6/HPifkdUag3QftSAror9eYhRNfKawBWFwtuDV6u6BGT2tIb3Duurq33/HJtkpv300s7T5ow9v1QwCA2R3N193++2gh5k5ouftdhATwmdeSEc6BESX3oL9eCqvRzVrhA3BPb+2DO6t2Hy1AlkvovuQ56+wefr8abZQevpc7tPugDQI6pxTWANzRQ1uh1W21cuwfnqrl380c/Xy086A95qBzSsENQA69thtK7sud8r7r9y5ZR7f8e9Vum7T83cSq/V0C/7KKO5cU3gDECJ07/dTxpEhZFbzEe++tPF6y5/zofVsyew96jr3ht68DtCM4oLdQIFGHAhyAWCEhstW2RUy9V3LOecxrzzTEfTFyUM/x2c6OUdsO2hUU0HsopClDAQ5AqpCh2DXbGEcrz9cM6DnUXHm+FS21X/a+05aOD2jLaUAfpZDmHpUIAKW12OboOajX2Le9NdovQI++1z4A2jdCJQ1Au9QzeeUIpsItQB0COpdU0gDksg3jLYbzH39+/1L3qf8Bajgc4t5ihUk9KmkAUp2t6t5Ce6P0glxPK1Fnt/T5zmivAL0T0Amm0gMgVg+rue8dV8pWZrU/x1qLq7ofLcp39/VoR8jCkMC5t+1ftFS5AAB966GhXiowjtKmCt2uq/Rq5T2cSzNb//4pN7iA/3zpQR+lIqEchSwAKc561lpqf+Ss57Yh5cnP2Wp9nfIdtPpZ+Cr0t/V7wrn/D+gtVY60SYEKQIqr3rRW2iBP1HNPfNZW6+uYz97qZ+CY3xfyeGulUqR9Z4v8AECPnq7Xnu5Jb8moUwCI/y0NgYdj/8xBBwCgjNLD3s8W4qsdhGI+6/p4t8+r/Tn4151z2cJy8NU3dy9JoRAFIIea7ZCW67LQhdlCX6vFz2pI9Lhir2u/L/ylB50krVb2ABAitQ57amiuOvaro3aH9kibWl5rAlqnB51kKkQA7mqtBz10uG3NObQxQ4J7C7BX58PZEP7Qx1KfOehwbIqAvjcvZuZFWnJTsAKQqvW6+KyOqxEKY4JNb+F8LTbAhZxHvX4XwFymCeg5X2+G7yyWSg+AFD3Vqa303obuKT9q3Xx0zlzdTBn1+wDG8fnz/cMc9B1nFW1PDQkAeNJIIWj9OUKGXW8fU/K7iLlRECMl+N59/ZjXvnN8o5yXwLiWMk5A3zgq/AVzABjfXpAL2bqstRv5oYE05HhzbYN1dCNj+5iY17tzPAAtKjrEvZXKKnahl9rH2yOVJMDcYoNWa3Vt7uHRNUYTpAxvT/kdYnu9t49PXQgOYFTrcvEwoOcI1zHDw1Jf/+5d3dYaCL1SmQKwCAmKLdW/rdVhd292pNxMiHl8zHsdnQv2yQb4azeg55gDdCZ2YZWz9yg5F4p7VKAAxKpRH4f06vZQp90N51evl+IqiKd0AvXwW8BsSnXA5n7NXtyag55z6Pr2tY5+lJl/LACgnJj1Z1prj6SOJix5UyT0eELak6193zCbp2+gjrTYaKylPP8WMgztSq97T+pBL6PF3xqAtvVWJx/1wNeqA1MbtTW/99CeduA5LZTFs177y3f/bf2HO19GTwt+tHDija6l3xuAOmK2xeq1bt6bltdzHfj073DUi97zdwgt6bFsnfX6/xLQ139Rau5SK190jydpz1r53QF4Vmp7YIR6erS6r9RvctSDPtr3B08boRx9veYqC9ajoL70oC9yL8LWypc7ysnak1Z+ewDqiQ1fvdbXI9d5pReDaq3NCD3ptcwMMUuZsA7ob69Xub2/W/pCRz5xWzbzQg8Aszkq82PrgVLtEtLF7mW+95yY545Ku4jcZr6etnq9vra/4bejB47EiVtXjxcKAPX1Un/P2BNcanu3Ub83Q/jJqZeyMYfY66XXkL4moFNF7xcOAM9ouQ6/WuxOXRdnhIb1kdFvQPCMlsvD0ka8do7KPAGdpo14MQIQp5V6PGaYt/qLGK0vskxZ66DWSnnXohmuh8+f7x/DB3Qn+RhmuCABRnXUSxDTq9jKft0hx6LOYnvOX53rAvoc5JJ0I1wLIaOEhg/oLoIxjXCBAswuZij43nznJ7b+CjmW2Oe3yLD8/FLOE7/DmOSRfHq+LkLD+es1wRB3F8X4er5YATgWshBZ6Xr+bNjpCPWPYfp5bUN2zPc78hz8GckgZfR2jYSMFLOKO0Pq7WIFII+Whr5v9RTqDbG+L/ZcPLr54zvvn+xRzijXx9k5MkVAf71cKDMY5YIFGE3OnsHW6vO9ecahj2/JWc9vq8ccI3YV9dTQnOP8HOH7pr2yaiQ9XyMh50XxgN7acB0Xy9haOtcASLPXdhil/g5ZJExdlleJkQwx80lj+f37tZwXo5RXLevxOgk9L7IH9B6Gc7loxtfaOQfAsdShwctzW28Qx67cffU8woXO/0zdSUBIZ9FyGTSqmtdJTCd07LmRLaD3uMiIC2lsrZ53AHwNTjH1ceq83bt1/lmPfmgAjx0OH3tMR3pso5VSou1393cNvTkw22/VA1libLHXXK7z4S3Hi1wdjEKFGpx3AG1atxtSwvnRa5Yo80OG2tdqpF99ZuHhP09+D2er/p89nr64tliUOBdu96DnqlhrcHGNr7VzDoD/hPaAb0PoneHwV8dw9bxR2w2j1pU1gnno+4/6nY9s1OufY7VugN4K6D2H84WLbXytnnsAxMs9hPhqe7EZ2gm915NXN3ue+g1Dp1n0/n3PYoZrn3hPrHmSFNBz3r1uwVnl7OIcR+vnIQD/KlUPnwXw3HPFW9dr/RjaQ13rN6wxJYNwM1zb9Ck6oI8WzhdHd1+F9/H0ck4CzK5kPbtXz4+6tduV3urFHn+X3r7jEfV43jCnqEXiRj6xc801m2lIHAD0bqa55ls9hsZefxs95s/r9VyBqB70mVakTBk2Neud916NcJ4CjCx3PWrF8//0Uv/thdpRfqeY3+BoocRefsfStm3xUc4R5hUc0K9O9pEKiasFY3K8Fu0a6VwGGE1qvTrbCLdtUKlVt+Xe732U3y/1xsPoQX2U3xfuCBriPts2EXuFpnA+rhHPYQDmtdRrteu3Eu2g2XtIj0YV1P6tU8z8O8KZoB70Xi/8GhQ2fXFeA/QndcHaGeroVuq1Gb7rGs5GE6xvXrRyHqw5JyBMUA96ixc5ADA+jXrYd7YTQQuda65dSBO1ijsAAOOKCXYW52rD3u9VO6g7FyBd9D7oXFMo9aP23WUArt2pV2ca3r5ICdixr7EEv/X/xx4n6Y4WmQv9+xyOFiF0LsA9AnoBCqa+COkAbUupV0fdnivW3e8hpI6c9bttwdXWv7lW8vcbw3ME9AwUWn0T0AHaF1PX6s376+53IaC3L/Y3jm33hL6m0RSQh4CeSOEzFiEdYGwj19tn841Lfu4Zpw+0JOfoiKve96Pn++0hPwE9kQJpPEI6wHNyDb29877Qq71wHbvN2vaGztU16dqBZwjoiRRS4xLUAco6qkNTyt87Q3rV5fSqZlvFdQNlCeiJFE5jE9IByqhZf1o4jlHZSg3G8b32AUCLVDoAY3HjlZF9/nz/0HaBMQjocEBFB5BXC+Vqrbnv8IQlqJe+1lq4lmFUhrhnoJBq353FTjTgAPJopb60ABYzCJ3Ssd4i7eo1XSdQnoCegcKqP6n7wgrrQE/u1k+5yrzW60n7psNXFlaEegT0mxRU/RLSgdal1DE59yYW0AGLK8KzBPRECqcxnA1zPPo3AR2IFVt2tFTHCOnA6yWow1ME9EQKpbEI3cBd23rhTrnSYh1z9nlCF15r8XMtzEuHa8t14tqAcgT0BAqlcQnqQIocK4P3ULeEhtir76D1z6qnEI4J6VCWgB5JYTQPYR14Ui/1S2x47XWxqdQpCTnXAIAWCehQloCeQIE0DyEdeL2eW3+i9frl7jDwUeZ5h54LPX9GOCOkQzkCeiIF0twEd+jT2T7ATx/L69VPXXI3kM7SmB/lBgRcmeWahhreah9Arwxhm5vV3OEZJbZALHHt5lwgjn6pG5iBcA5l6UFPsN6GS+E0Nw0xuO9sgbUWg29ouR9yrL3WIbFD3dWXMA4BHcrKGtBzNlpac7RKrcKJHs9nqOEqwD15LIv1Ddft3+XU60JpZ2K+Mw16GIepHFBWloAee3H2FGhCGh0KJ3o6p+EJT4fxHsrhmbfuEtBhLNq/UE62HvTZQ3rI4xhXT+czlLbXK13qPXoioM/1mQEgRZUe9EVq4y1X4+9o2HrM4/eetzdnUqNkfEI6lDVCORpbTozwmV8vAR0AQiUH9JKVbGpILsVcG0IJ6cyk5IrVs5W1I85RX3OjGgDC3O5Bz13hxq4M+xR3/wklpEO6mcvYVuu/HAR0AAhTZRX3I3qqGYWQzmjOtkLL/fqMR0AHgDDZ90FXAYOAzlhKDmVfXr/Ua9MGo9AAIMz32gcAQNvccOKu9eKuy/9qHxMAtCh7QF9XvEcVsIoZoA+fP98/SvZ66lGdR+lzCQBGkH2I+xGVMjNxE4rexW5Dmes9AABm9tgQd4GFGRi6yQieCM7COQDAv95qHwCMQjBnBEfB+e75LZADAFx7LKBrnDEioZwZ3DnPlf0AAOGKB3SNM0YkmDMDwRwA4FnFFonTOGNUwjmcU/4DAKQpEtA1zhhVjnBe6vpw44DalP0AAPfcCugpjbG9EPH58/1j/fcaebTkx5/fv7bnaIrWzmuBnj3b8/SozH7uiAAA5pEU0GMaZ6EhQIOPnsSG25bPb0GdRcvnKQDADKIDekgDLrXBr3FIT2LO8x7ObUGdHs5TAICRfY95cMlwDr0JuR4+f75/9BJ6ejpWAAAYUXAP+l7DfW/euN5z+DtvvfZx3OFm21x6P18BAEZQbJu1FBqI0B5BfQ7KXwCA+poK6It1b7xGI7RBUB+XchYAoA1NBvSFRiO0RUgfk7IWAKANUYvEPUmDEdrjuhyP3xQAoB1vtQ9gS2MRAACAGTXVgy6cQ/tcpwAAUEZTAR0AAABm1eQicXrooF0WihuHshYAoC1NBvQtjUhoi5A+BmUrAEBbmlskbqHhCAAAwEzMQQeiuYHWP78hAEB7mgzoGo7QNkPc+6aMBQBoU5MBHWiXcA4AAGU0F9D17ACUo4wFAGhXUwFdwxHapvccAADKaSqgA1COm6AAAG1rJqBrOELb9J73TRkLANC+JgK6hiO0TTgHAIDyvtV4U4Ec+iGc9015CwDQj7cn3kQDEeB5yl4AgL4UDegahwB1KH8BAPpTZIi7hiGMwfD2PimDAQD6lNyDrgEI0B5lMwBAv6ICuoYfQJuUzwAA/bsc4q7RB/MwpL0vymcAgLEcBnQNP5iHYN4fZTQAwHi+ff58/1ga5xp8MB/hvD/KagCAMX3T0IN5CeftU0YDAMzje+0DAPL78ef3r234Xv95799py+fP9491OPd7AQCMTw86TEC465tyGgBgDsn7oAPtE8z7J5wDAMxDDzpMRGDvk3IaAGAOetBhAoJ5P7Zh3G8HADAPPejQqR9/fv8KuX4FvPYphwEAeL12etBDG/0h7K8O+ewFbeEbAADG8U9AjwnTMeEgZ/CH0R1dW8s1JJj3SzkIAMCR6DnoKcFAOIc0rptx+C0BALjy/cef37+e6I3T43fMd8PW58/3D4EOAADm8vZkCDgLojOGEcEc5mFNDgAAruwOcX8iOH7+fP+YOaCuP7sGOzFmvm5651oHAODMt6ffMLaBehVGemvw6kXjLgG9fa5vAABSPB7QX6/9xuud0NFyY3jvc7V8vLRLMG+b6xoAgLuqBPQSajWOU0KThjxrZ9MdhPI+uKYBAMhBQE+UGpw05Fm2HRS+x+LaBgDgrmEC+utVvoGst5wcBPNxud4BALhjqIC+J0eDOTZQaaQTY71w4HYRQWF+PFflw/Y3V54AAMxj+ID+et1v4MaEJI1pQqUEMYF9HLGLZSpbAADGN0VAP7Nu9I66kjztWuajpzwv9T1z76Iws9je8JTXDnkN5Q8AwBimD+g5aBxTynbI+/rvFjWmcczgKCDf+b7vhu298wEAgHEI6BloLNMz4Xyf6xoAgKcJ6DdowNMzwfwv1zIAAC0Q0G/QqKd3M4d01y8AAK3pPqDnWuQtx/tDb2YM6K5ZAABa1VRAf6rhnCOUaOQzgpkCumsWAIDWVQ/oGs1Qz6gBXbkCAECPqgV0DWioZ8RgrkwBAKB3VXvQNajhWaMFc2UIAAAjeat9AEA5owXy10soBwBgXNXnoL9eGtyQw4hhfKGMAABgBlkD+rYRfRQYNLbhvl4DuesfAAD2FR3iriEO83HdAwBAGnPQgWTCOAAA5JMtoGuowxxc6wAAUIYedGhMy3PLhXMAAChHQOe2JVAKb/GEcQAAYJFtFXeNeX78+f3r6DzYC/Fnjx9Ny0H8zCy/DwAAtEAPOlmtg+g63H3+fP9Y/9vy30tI3/5/yeO6kuP9ew3kAABAPQI6tx2F0ZCh76k96iUD8Pa1YwO7cA4AAKRIHuJu6Ot4rgL1UfA8OxfuhtVtzzvPcp0DAMBzkgK6RvtXVwHy6e9r73iWY8gRdu98HmG7H65zAAB4VnRAn7HRnjNUPvn9tXrcQnofZrzWAQCgpuCAPlpjPWTF8dJmC+uCeV9Gu+YBAKB130MeNFND/ckQ+ePP719X75freHL3gMccV+zjAQAAZvRPD/osYXz0wFhrcbXtquxPvz/5zFIWAABAK74E9Fka5D0ER6uX04JZygQAAGjB22wNcKEXwsXsTQ8AANzzVvsA2OdGArUJ5gAA8KygReIAAACAsqoH9Kd6iq0kDgAAQMuqB3TDaKE9rksAAHhe9YD+lM+f7x9CB4Qx2gQAAJ73rWRo3Wvkb99v/ZjSAVrogDhuagEAwHOCA/qT4TZXKBDIIY1gDgAAzwvaZu3poHv2fi3eUAAAAIC7upuDbjV2AAAARnQZ0IVhmI8bYQAA8Lxvnz/fPzTEgdfreArJjz+/f5mXDgAAZXU3xB0o4yyAC+cAAFDed73nAAAAUJ8edEAPOQAANOBb7QMA6hDKAQCgLQI6TEYwBwCANgnoMAnBHAAA2vZ/grbcympFUEAAAAAASUVORK5CYII=";

// ── Colour config ─────────────────────────────────────────────────────────────
const TYPE_COLOR = {
  0: "#64748b",  // Unknown
  1: "#22c55e",  // Cargo
  2: "#ef4444",  // Tanker
  3: "#a855f7",  // Passenger
  4: "#f97316",  // Fishing
  5: "#06b6d4",  // Tug/Service
  6: "#f59e0b",  // Pleasure/Sail
  7: "#94a3b8",  // Other
};
const GROUP_NAMES = [
  "Unknown","Cargo","Tanker","Passenger",
  "Fishing","Tug/Service","Pleasure/Sail","Other",
];

// ── Replay state ───────────────────────────────────────────────────────────────
let ships       = [];
let markers     = [];
let currentTick = 0;
let playing     = true;
let REPLAY_MODEL = "transformer";   // which model's predicted path Replay draws
let ACTIVE_MODEL = "transformer";   // model currently running (Replay draw / Sim inference)
const MODEL_NAMES = { gru: "GRU", tcn: "TCN", transformer: "Transformer" };
// Phase-badge suffix naming the model currently being run.
function _modelTag() {
  const n = MODEL_NAMES[ACTIVE_MODEL] || ACTIVE_MODEL || "";
  return n ? ` · ${n}` : "";
}

// Anomaly level / shown σ for a vessel UNDER the selected Replay model. precompute
// emits per-model maps (model_levels/model_sigma) keyed to each model's scoreline
// σ; fall back to the legacy transformer fields if an older ships.json lacks them.
function _shipLevel(s) {
  return (s.model_levels && s.model_levels[REPLAY_MODEL]) || s.level;
}
function _shipSigma(s) {
  const v = s.model_sigma && s.model_sigma[REPLAY_MODEL];
  return v != null ? v : s.z_peak;
}
let speedMult   = 2;
let rafId       = null;
let lastTime    = null;
let msPerTick   = 200;
let SEQ_ENC     = 60;
let SEQ_DEC     = 10;
let WIN_LEN     = 70;
let N_FUTURE    = 10;
let PREDICT_EVERY = 5;
let REPLAY_THRESH = null;
let ISSUED_STEPS  = [];          // global — same stride for every ship
let replayEnded   = false;
let SCOREBOARD_MODELS = [];      // e.g. ["gru","tcn","transformer"] — display order
let MODEL_SCORES      = null;    // fleet-wide aggregate {model:{ade,fde,n}} (final)
let DETECTION_BY_TYPE  = null;   // SET 1 per model: {model: {type_name: {precision,recall,f1,opt_sigma,...}}}
let DETECTION_COMBINED = null;   // SET 2 per model: {model: {precision,recall,f1,false_alarm_rate,n_injected,n_real,...}}
let REAL_DETECTION     = null;   // per model: {model: {n_real, flagged, recall}} — real FLAGS faults
let activeTypes  = new Set(Object.keys(TYPE_COLOR).map(Number));
let injectedOnly = false;   // Replay: when on, show only injected vessels
let activeLevels = new Set(["none","mild","moderate","severe"]);

// ── Sidebar state ──────────────────────────────────────────────────────────────
let _sidebarOpen = false;
function toggleSidebar() {
  _sidebarOpen = !_sidebarOpen;
  document.getElementById("sidebar").classList.toggle("open", _sidebarOpen);
}

// Right-hand anomaly panel: a slide drawer like the left sidebar. One shared
// open/closed state drives whichever panel belongs to the current mode —
// #sim-feed in Live Sim, the Top-Anomalies #anomaly-panel in Replay — so the
// Alerts button always toggles the visible one (and Replay's is closeable too).
let _anomOpen = true;
function _setAnomPanel(open) {
  _anomOpen = open;
  document.getElementById("sim-feed").classList.toggle("open", open && simMode);
  document.getElementById("anomaly-panel").classList.toggle("open", open && !simMode);
  document.getElementById("btn-alerts").classList.toggle("active", open);
}
function toggleAnomPanel() { _setAnomPanel(!_anomOpen); }

// ── Map setup ─────────────────────────────────────────────────────────────────
const BOUNDS = L.latLngBounds([48.0, -7.0], [68.0, 22.0]);
const map = L.map("map", {
  zoomControl: false, maxBounds: BOUNDS, maxBoundsViscosity: 1.0,
  minZoom: 5, maxZoom: 14,
}).setView([58.5, 8.0], 6);
L.control.zoom({ position: "bottomright" }).addTo(map);

let _tileLayer = L.tileLayer(
  "https://{s}.basemaps.cartocdn.com/rastertiles/voyager/{z}/{x}/{y}{r}.png",
  { attribution: '&copy; <a href="https://carto.com">CARTO</a>', subdomains: "abcd", maxZoom: 19 }
).addTo(map);

// The model's training land-penalty raster, as a semi-transparent overlay.
// Bounds = the region the raster covers (LAT 50–66, LON -5–20); the PNG is
// north-up (flipped from the array's south-up storage).
const PENALTY_BOUNDS = [[50.0, -5.0], [66.0, 20.0]];
let _penaltyOverlay = null;
let _penaltyVisible = false;

function togglePenaltyMap() {
  _penaltyVisible = !_penaltyVisible;
  if (_penaltyVisible) {
    if (!_penaltyOverlay) {
      // Prefer the inlined data URI (can never 404 on deploy); fall back to file.
      const src = LAND_PENALTY_PNG;
      _penaltyOverlay = L.imageOverlay(src, PENALTY_BOUNDS, {
        opacity: 0.8, interactive: false, className: "penalty-overlay",
      });
      _penaltyOverlay.on("error", () =>
        console.error("penalty overlay image failed to load"));
    }
    _penaltyOverlay.addTo(map);
    _penaltyOverlay.bringToFront();   // sit above the base tiles, below vessels
  } else if (_penaltyOverlay) {
    map.removeLayer(_penaltyOverlay);
  }
  document.getElementById("btn-land").classList.toggle("active", _penaltyVisible);
}

// ── Load static data ──────────────────────────────────────────────────────────
fetch("/api/ships", { cache: "no-store" })   // always pick up a freshly-precomputed set
  .then(r => r.json())
  .then(data => {
    document.getElementById("loading").classList.add("hidden");
    ships = data.ships || [];
    if (data.seq_enc)    SEQ_ENC  = data.seq_enc;
    if (data.seq_dec)    SEQ_DEC  = data.seq_dec;
    if (data.window_len) WIN_LEN  = data.window_len;
    else WIN_LEN = SEQ_ENC + SEQ_DEC;
    N_FUTURE      = data.n_future_steps || (WIN_LEN - SEQ_ENC);
    PREDICT_EVERY = data.predict_every  || 5;
    REPLAY_THRESH = data.anomaly_threshold || null;
    // Flagged when the per-ship-type-calibrated deviation crosses the threshold,
    // held for `persistence` consecutive steps.
    if (REPLAY_THRESH != null) {
      const t = document.getElementById("ap-thresh-note");
      const parts = [`≥ ${REPLAY_THRESH.toFixed(1)}× class median deviation`];
      if (data.persistence > 1)     parts.push(`${data.persistence}-step`);
      if (t) t.textContent = "flagged at " + parts.join(" · ");
    }

    // Backward compatibility: old ships.json has a single pred/actual pair.
    ships.forEach(s => {
      if (!s.predictions) {
        s.predictions = [{ pred_id: 0, issued_step: SEQ_ENC,
                           coords: s.pred, sogs: [], ade_km: s.ade_km }];
      }
      if (!s.future) s.future = s.actual;
    });
    ISSUED_STEPS = ships.length ? ships[0].predictions.map(p => p.issued_step) : [];
    SCOREBOARD_MODELS = data.scoreboard_models || [];
    MODEL_SCORES      = data.model_scores || null;
    DETECTION_BY_TYPE  = data.detection_by_type || null;
    DETECTION_COMBINED = data.detection_combined || null;
    REAL_DETECTION     = data.real_detection || null;

    const tb = document.getElementById("tick-bar");
    tb.max = WIN_LEN - 1;
    document.getElementById("st-tick").textContent = `0 / ${WIN_LEN - 1}`;
    buildTickMarks();
    buildLegend(data.type_counts);
    buildAnomalyList();
    buildReplayFilter();
    renderDetectionMetrics();
    updateStats();
    createMarkers();
    updateTick(0);
    startLoop();
    _setAnomPanel(true);          // Top-Anomalies panel open by default (replay)
  })
  .catch(() => {
    document.querySelector("#loading-msg").textContent =
      "No precomputed data — use Live Sim mode or run precompute.py.";
    document.querySelector(".spinner").style.display = "none";
    setTimeout(() => document.getElementById("loading").classList.add("hidden"), 2500);
  });

// Ticks on the replay timeline marking where each overlapping prediction is issued.
function buildTickMarks() {
  const wrap = document.getElementById("tick-marks");
  if (!wrap) return;
  wrap.innerHTML = "";
  ISSUED_STEPS.forEach(s => {
    const el = document.createElement("div");
    el.className = "tick-mark";
    el.style.left = (s / (WIN_LEN - 1) * 100) + "%";
    el.title = `Prediction issued at step ${s}`;
    wrap.appendChild(el);
  });
}

// ── Legend ────────────────────────────────────────────────────────────────────
function buildLegend(typeCounts) {
  const grid = document.getElementById("type-legend");
  grid.innerHTML = "";
  GROUP_NAMES.forEach((name, idx) => {
    if (!typeCounts || !typeCounts[name]) return;
    const item = document.createElement("div");
    item.className    = "type-chip";
    item.dataset.type = idx;
    item.innerHTML = `
      <span class="type-dot" style="background:${TYPE_COLOR[idx]}"></span>
      <span class="type-name">${name}</span>
      <span class="type-count">${typeCounts[name]}</span>`;
    item.onclick = () => toggleType(idx);
    grid.appendChild(item);
  });
  // "Injected only" toggle: show its count, hide the control if nothing injected.
  const injEl = document.getElementById("injected-filter");
  if (injEl) {
    const n = ships.filter(s => s.injected).length;
    injEl.style.display = n ? "" : "none";
    const c = document.getElementById("inj-count"); if (c) c.textContent = n;
  }
}

// Replay anomalies stream into the panel as the replay clock passes the moment
// each one peaks (trip_tick) — rather than all appearing up-front — mirroring
// the Live Sim feed. Clicking one jumps to where/when it occurred.
const _anomShownIds = new Set();

function buildAnomalyList() {   // reset to empty; entries reveal via updateAnomalyFeed()
  const c = document.getElementById("anomaly-entries");
  if (c) c.innerHTML = "";
  _anomShownIds.clear();
}

function updateAnomalyFeed(tick, rebuild) {
  const container = document.getElementById("anomaly-entries");
  if (!container) return;
  if (rebuild) { container.innerHTML = ""; _anomShownIds.clear(); }

  // A vessel is revealed once the clock passes its earliest active alert.
  const cand = [];
  ships.forEach(s => {
    if (_anomShownIds.has(s.id)) return;
    const al = _vesselAlerts(s).filter(a => _activeReplayKinds.has(a.kind));
    if (!al.length) return;
    const revealTick = Math.min(...al.map(a => a.tick));
    if (tick >= revealTick) cand.push({ s, al, revealTick });
  });

  // Oldest first so the newest ends up on top after prepending (and flashes in).
  cand.sort((a, b) => a.revealTick - b.revealTick).forEach(({ s, al }) => {
    _anomShownIds.add(s.id);
    const kinds = [...new Set(al.map(a => a.kind))];
    const chips = kinds.map(k => {
      const m = REPLAY_KINDS[k];
      return `<span class="feed-chip" style="--c:${m.col}">${m.label}</span>`;
    }).join("");
    const sv = _shipSigma(s);
    const z = sv != null ? `${sv.toFixed(1)}σ` : `${s.ade_km.toFixed(1)} km`;
    const injTag = s.injected
      ? ` <span class="mmsi-tag" title="Injected synthetic anomaly">INJ</span>` : "";
    const faultTag = s.real_anomaly
      ? ` <span class="mmsi-tag" style="color:#c0a080;border-color:#b0896855" title="Real data-integrity fault (FLAGS)">FAULT</span>` : "";
    const div = document.createElement("div");
    div.className = "feed-entry";
    div.style.borderLeftColor = REPLAY_KINDS[kinds[0]].col;
    div.onclick = () => jumpToReplayAnomaly(s.id);
    div.innerHTML =
      `<span class="feed-mmsi">${s.type_name}${injTag}${faultTag} ${s.mmsi || "#" + s.id}</span> <span class="feed-time">${z}</span>
       <div class="feed-chips">${chips}</div>
       <span class="feed-reason">${al[0].reason}</span>`;
    container.insertBefore(div, container.firstChild);
  });

  while (container.children.length > 20) container.removeChild(container.lastChild);
}

// ── Replay anomaly types + type filter ────────────────────────────────────────
// Model-deviation subtypes (speed/trajectory) PLUS the physical rule
// detectors ported from the Live Sim (on_land/speed_jump/loitering).
const REPLAY_KINDS = {
  on_land:     { col: "#d98a4e", label: "On land" },
  speed_jump:  { col: "#e0554e", label: "Position jump" },
  loitering:   { col: "#8ea0ac", label: "Loitering" },
  speed:       { col: "#59b39a", label: "Speed anomaly" },
  trajectory:  { col: "#5f95b5", label: "Off predicted path" },
};
Object.values(REPLAY_KINDS).forEach(m => { m.txt = _txtOn(m.col); });
const _activeReplayKinds = new Set(Object.keys(REPLAY_KINDS));

// The model-deviation subtype for a flagged vessel (speed / trajectory). A sharp
// heading change is no longer its own kind — it reads as "off predicted path"
// (trajectory), with the turn detail kept in the reason text.
// position-jump / loitering are handled by the rule detectors, not the model.
function _replayKind(s) {
  if (s.injected && s.anomaly_type) {
    return ({ speed_drop: "speed", speed_surge: "speed" })[s.anomaly_type] || "trajectory";
  }
  const r = (s.reason || "").toLowerCase();
  if (r.includes("deceler") || r.includes("acceler") || r.includes("low-speed")) return "speed";
  return "trajectory";
}

// Combined alerts for a Replay vessel: the model-deviation anomaly (if flagged
// severe/moderate) plus each physical rule-detector alert. Memoised on the ship.
function _vesselAlerts(s) {
  // Cache is keyed by model — the model-deviation alert depends on REPLAY_MODEL.
  if (s._alertsModel === REPLAY_MODEL && s._alerts) return s._alerts;
  const out = [];
  // Flagged ⇔ deviation reaches the selected model's per-class σ (level != none),
  // the same test the scoreline uses; the physical rules are model-independent.
  if (_shipLevel(s) !== "none" && s.trip_tick != null) {
    out.push({ kind: _replayKind(s), tick: s.trip_tick, lat: s.trip_lat, lon: s.trip_lon,
               reason: (s.reason || "").split(";")[0] });
  }
  (s.rule_alerts || []).forEach(a => out.push(a));
  s._alerts = out;
  s._alertsModel = REPLAY_MODEL;
  return out;
}

function buildReplayFilter() {
  const wrap = document.getElementById("replay-anom-filter");
  if (!wrap) return;
  const counts = {};   // distinct vessels carrying each alert kind
  ships.forEach(s => {
    new Set(_vesselAlerts(s).map(a => a.kind)).forEach(k => counts[k] = (counts[k] || 0) + 1);
  });
  const chip = ([k, m]) => `<span class="anom-chip${_activeReplayKinds.has(k) ? " active" : ""}" data-rkind="${k}" style="--c:${m.col};--tc:${m.txt}"
         onclick="toggleReplayKind('${k}')"><span class="chip-dot"></span><span>${m.label}</span><span class="ac-n">${counts[k]}</span></span>`;
  wrap.innerHTML = Object.entries(REPLAY_KINDS)
    .filter(([k]) => counts[k])
    .map(chip).join("");
}

function toggleReplayKind(k) {
  if (_activeReplayKinds.has(k)) _activeReplayKinds.delete(k); else _activeReplayKinds.add(k);
  document.querySelectorAll(`.anom-chip[data-rkind="${k}"]`)
    .forEach(el => el.classList.toggle("active", _activeReplayKinds.has(k)));
  updateAnomalyFeed(currentTick, true);   // rebuild with the new filter
}

// ── Vessel icon: AIS triangle for known types, circle for unknown (type 0) ───
function _vesselIcon(color, cogDeg, level, isSelected, shipType, injected) {
  const sizes = { severe: 14, moderate: 13, mild: 12, none: 11 };
  const s  = isSelected ? 15 : (injected ? 14 : (sizes[level] || 11));
  // Injected vessels always carry a distinct purple outline so they're
  // identifiable regardless of whether the detector flagged them.
  const sc = isSelected          ? "#fff"     :
             injected            ? "#f3ca86" :
             level === "severe"   ? "#ef4444" :
             level === "moderate" ? "#f97316" :
             level === "mild"     ? "#f59e0b" : "rgba(30,30,30,0.35)";
  const sw = (isSelected || injected || level !== "none") ? 1.8 : 0.6;

  if (shipType === 0) {
    // Unknown vessels: plain circle, no rotation
    return L.divIcon({
      html: `<svg width="${s}" height="${s}" viewBox="0 0 10 10" xmlns="http://www.w3.org/2000/svg">
               <circle cx="5" cy="5" r="4" fill="${color}" stroke="${sc}" stroke-width="${sw}"/>
             </svg>`,
      iconSize: [s, s],
      iconAnchor: [s / 2, s / 2],
      className: "vessel-icon",
    });
  }

  // AIS-style compact triangle with stern notch, bow pointing up (north = 0°)
  const h = Math.round(s * 1.2);
  return L.divIcon({
    html: `<div class="vi-rot" style="transform:rotate(${cogDeg}deg);width:${s}px;height:${h}px">
             <svg width="${s}" height="${h}" viewBox="0 0 10 12" xmlns="http://www.w3.org/2000/svg">
               <path d="M5,0 L10,12 L5,8.5 L0,12 Z"
                     fill="${color}" stroke="${sc}" stroke-width="${sw}"/>
             </svg>
           </div>`,
    iconSize: [s, h],
    iconAnchor: [s / 2, h / 2],
    className: "vessel-icon",
  });
}

function _shipCog(enc) {
  if (!enc || enc.length < 2) return 0;
  const dLat = enc[enc.length - 1][0] - enc[0][0];
  const dLon = enc[enc.length - 1][1] - enc[0][1];
  return (Math.atan2(dLon, dLat) * 180 / Math.PI + 360) % 360;
}

// COG at a specific replay tick — bearing to the next point in the track.
function _tickCog(ship, tick) {
  let a, b;
  if (tick < SEQ_ENC - 1) {
    a = ship.enc[tick];
    b = ship.enc[tick + 1];
  } else if (tick === SEQ_ENC - 1) {
    a = ship.enc[tick];
    b = (ship.future && ship.future[0]) || (ship.pred && ship.pred[0]);
  } else {
    const ft = tick - SEQ_ENC;
    a = ship.future && ship.future[ft];
    b = ship.future && ship.future[ft + 1];
  }
  if (!a || !b) return _shipCog(ship.enc);
  const dLat = b[0] - a[0];
  const dLon = b[1] - a[1];
  if (Math.abs(dLat) < 1e-9 && Math.abs(dLon) < 1e-9) return _shipCog(ship.enc);
  return (Math.atan2(dLon, dLat) * 180 / Math.PI + 360) % 360;
}

// The predicted path for a Replay prediction under the currently-selected model.
// Transformer's path is stored in `coords`; GRU/TCN paths live in `model_coords`.
function _predCoords(pred) {
  return (pred.model_coords && pred.model_coords[REPLAY_MODEL]) || pred.coords;
}

// ── Marker creation (replay mode) ─────────────────────────────────────────────
function createMarkers() {
  markers.forEach(m => {
    if (map.hasLayer(m.marker)) map.removeLayer(m.marker);
    map.removeLayer(m.trail); map.removeLayer(m.gtLine);
    m.predLines.forEach(pl => map.removeLayer(pl.line));
  });
  markers = ships.map(ship => {
    const col  = TYPE_COLOR[ship.type] || "#64748b";
    const cog  = _shipCog(ship.enc);
    const icon = _vesselIcon(col, cog, _shipLevel(ship), false, ship.type, ship.injected);
    const marker = L.marker([ship.enc[0][0], ship.enc[0][1]], { icon }).addTo(map);
    marker.bindTooltip(makeTooltip(ship), { sticky: true, opacity: 0.92 });
    marker.on("click", e => { L.DomEvent.stopPropagation(e); selectReplayShip(ship.id); });
    const trail  = L.polyline([], { color: col, weight: 1.8, opacity: 0.65 }).addTo(map);
    const gtLine = L.polyline([], { color: col, weight: 2, opacity: 0.55 }).addTo(map);
    // One dashed polyline per overlapping prediction, hidden until the replay
    const predLines = ship.predictions.map(p => ({
      pred: p,
      line: L.polyline([], { color: col, weight: 1.5, opacity: 0.35, dashArray: "6 5" }),
      shown: false,
    }));
    return { marker, trail, gtLine, predLines, _vis: true };
  });
}

function makeTooltip(ship) {
  const sog = ship.sogs ? ship.sogs[ship.sogs.length - 1].toFixed(1) : "—";
  const levelBadge = {
    none:     '<span style="color:#64748b">Normal</span>',
    mild:     '<span style="color:#f59e0b">Mild anomaly</span>',
    moderate: '<span style="color:#f97316">Moderate anomaly</span>',
    severe:   '<span style="color:#ef4444">&#9888; Severe anomaly</span>',
  }[_shipLevel(ship)];
  const injTag = ship.injected
    ? `<br><span style="color:#f3ca86">&#128137; INJECTED — ${ship.anomaly_type.replace(/_/g," ")} (${ship.severity} ${ship.severity_unit})</span>`
    : "";
  const mmsiTag = ship.mmsi ? ` <span style="color:#64748b;font-weight:normal">· MMSI ${ship.mmsi}</span>` : "";
  return `<b>${ship.type_name}</b>${mmsiTag}<br>SOG: ${sog} kn &nbsp; ADE: ${ship.ade_km.toFixed(1)} km<br>${levelBadge}${injTag}<br>
          <small style="color:#475569">${ship.reason.split(";")[0]}</small>`;
}

// ── Prediction reveal / restyle helpers (replay) ──────────────────────────────

// Flash the SVG element of a polyline briefly — used the instant a prediction
// "arrives" (replay reveal and live-sim SSE arrival share this treatment).
function flashLine(line) {
  const el = line.getElement && line.getElement();
  if (!el) return;
  el.classList.add("pred-flash");
  setTimeout(() => el.classList.remove("pred-flash"), 1100);
}

// Newest revealed prediction is dominant; older overlapping ones fade with age.
function _restyleReplayPreds(m, isSel) {
  const shown = m.predLines.filter(pl => pl.shown);
  const n = shown.length;
  shown.forEach((pl, i) => {
    const age    = n - 1 - i;
    const newest = age === 0;
    const op = newest ? (isSel ? 1.0 : 0.85)
                      : Math.max(0.12, (isSel ? 0.55 : 0.4) - age * 0.09);
    const w  = newest ? (isSel ? 3.5 : 2.4) : (isSel ? 2 : 1.3);
    pl.line.setStyle({ opacity: op, weight: w });
  });
}

let _predLabelTimer = null;
function _showPredIssuedLabel(k, step) {
  const el = document.getElementById("replay-pred-label");
  if (!el) return;
  el.textContent = `Prediction #${k + 1} issued at step ${step}`;
  el.classList.add("show");
  clearTimeout(_predLabelTimer);
  _predLabelTimer = setTimeout(() => el.classList.remove("show"), 1800);
}

// ── Tick update ───────────────────────────────────────────────────────────────
const TRAIL_LEN = 18;

function updateTick(tick) {
  const prevTick = currentTick;
  currentTick = tick;
  document.getElementById("tick-bar").value = tick;
  document.getElementById("st-tick").textContent = `${tick} / ${WIN_LEN - 1}`;

  const inDecoder = tick >= SEQ_ENC;
  const atEnd     = tick >= WIN_LEN - 1;
  const phase     = document.getElementById("phase-badge");
  if (!simMode) {
    phase.textContent = (atEnd ? "⬤ COMPLETE" : inDecoder ? "⬤ PREDICTING" : "⬤ HISTORY") + _modelTag();
    phase.className   = inDecoder ? "decoder" : "encoder";
  }

  // Global "prediction issued" pulse — every ship predicts on the same stride.
  if (!simMode && tick > prevTick) {
    const k = ISSUED_STEPS.findIndex(s => s > prevTick && s <= tick);
    if (k !== -1) _showPredIssuedLabel(k, ISSUED_STEPS[k]);
  }

  ships.forEach((ship, i) => {
    const m = markers[i];
    if (!m) return;
    const vis = activeTypes.has(ship.type) && activeLevels.has(_shipLevel(ship))
                && (!injectedOnly || ship.injected);
    if (vis !== m._vis) {
      m._vis = vis;
      if (vis) m.marker.addTo(map); else map.removeLayer(m.marker);
    }
    if (!vis) {
      m.trail.setLatLngs([]); m.gtLine.setLatLngs([]);
      m.predLines.forEach(pl => {
        if (pl.shown) { map.removeLayer(pl.line); pl.shown = false; }
      });
      return;
    }

    if (tick < SEQ_ENC) {
      m.marker.setLatLng(ship.enc[tick]);
      const start = Math.max(0, tick - TRAIL_LEN + 1);
      m.trail.setLatLngs(ship.enc.slice(start, tick + 1));
      m.gtLine.setLatLngs([]);
    } else {
      const ft = Math.min(tick - SEQ_ENC, ship.future.length - 1);
      m.marker.setLatLng(ship.future[ft] || ship.enc[SEQ_ENC - 1]);
      m.trail.setLatLngs(ship.enc.slice(SEQ_ENC - TRAIL_LEN));
      m.gtLine.setLatLngs([ship.enc[SEQ_ENC - 1], ...ship.future.slice(0, ft + 1)]);
    }

    // Reveal/hide overlapping predictions relative to the current tick.
    // Idempotent so scrubbing backwards works; flash only on forward reveal.
    let changed = false;
    m.predLines.forEach(pl => {
      const visible = tick >= pl.pred.issued_step;
      if (visible && !pl.shown) {
        const originIdx = pl.pred.issued_step - SEQ_ENC;   // index into future; 0 → last enc pt
        const origin = originIdx <= 0 ? ship.enc[SEQ_ENC - 1] : ship.future[originIdx - 1];
        pl.line.setLatLngs([origin, ..._predCoords(pl.pred)]);
        pl.line.addTo(map);
        pl.shown = true;
        changed = true;
        if (tick > prevTick && tick - pl.pred.issued_step <= Math.max(1, speedMult)) {
          flashLine(pl.line);
        }
      } else if (!visible && pl.shown) {
        map.removeLayer(pl.line);
        pl.shown = false;
        changed = true;
      }
    });
    if (changed) _restyleReplayPreds(m, selectedShipId === ship.id);

    // Update icon rotation to reflect current direction of travel
    if (ship.type !== 0) {
      const el = m.marker.getElement();
      if (el) {
        const rot = el.querySelector('.vi-rot');
        if (rot) rot.style.transform = `rotate(${Math.round(_tickCog(ship, tick))}deg)`;
      }
    }
  });

  if (selectedShipId !== null && !simMode) refreshReplaySelPanel(selectedShipId);

  updateScoreboard(tick);
  updateAnomalyFeed(tick, tick < prevTick);   // scrubbing back rebuilds the list

  // Play through once, then hold on the final frame (no auto-loop).
  if (atEnd && playing && !simMode) {
    playing = false;
    replayEnded = true;
    const btn = document.getElementById("btn-play");
    btn.textContent = "↺ Replay";
    btn.classList.remove("active");
  }
}

// ── Live accuracy scoreboard ──────────────────────────────────────────────────
// Running per-model ADE/FDE, measured against ground truth, over every
// prediction whose full SEQ_DEC-step horizon has been revealed by `tick`.
// Recomputed from scratch each tick so scrubbing/seeking stays correct.
function updateScoreboard(tick) {
  if (!SCOREBOARD_MODELS.length) return;

  const sum = {};
  SCOREBOARD_MODELS.forEach(mk => { sum[mk] = { ade: 0, fde: 0 }; });
  let n = 0, excluded = 0;

  for (const s of ships) {
    // Severe-anomaly, injected (tampered) and real-fault vessels are left out so
    // the board reads as normal-behaviour accuracy — same rule precompute applies.
    if (s.level === "severe" || s.injected || s.real_anomaly) { excluded++; continue; }
    for (const p of s.predictions) {
      if (!p.model_ade) continue;
      // "completed" once the vessel has actually travelled the full horizon.
      if (tick >= p.issued_step + SEQ_DEC - 1) {
        SCOREBOARD_MODELS.forEach(mk => {
          sum[mk].ade += p.model_ade[mk];
          sum[mk].fde += p.model_fde[mk];
        });
        n++;
      }
    }
  }

  // Until the live sample builds up (a prediction only "completes" once its
  // full horizon has been replayed, ~2/3 through the window), fall back to the
  // precomputed full-run aggregate so the bars are never empty when the panel
  // is open. Once live predictions complete, show the running measurement.
  const ade = {}, fde = {};
  let noteN;
  if (n) {
    SCOREBOARD_MODELS.forEach(mk => { ade[mk] = sum[mk].ade / n; fde[mk] = sum[mk].fde / n; });
    noteN = `live · n=${n.toLocaleString()}`;
  } else if (MODEL_SCORES && MODEL_SCORES[SCOREBOARD_MODELS[0]] &&
             MODEL_SCORES[SCOREBOARD_MODELS[0]].ade != null) {
    SCOREBOARD_MODELS.forEach(mk => { ade[mk] = MODEL_SCORES[mk].ade; fde[mk] = MODEL_SCORES[mk].fde; });
    noteN = `full run · n=${(MODEL_SCORES[SCOREBOARD_MODELS[0]].n || 0).toLocaleString()}`;
  } else {
    SCOREBOARD_MODELS.forEach(mk => _setScoreBar(mk, null, null, 0, 1, 0, 1, false, false));
    noteN = "no data";
  }

  const nEl = document.getElementById("pp-score-n");
  if (nEl) nEl.textContent = `· ${noteN}${excluded ? ` · ${excluded} excluded` : ""}`;

  if (ade[SCOREBOARD_MODELS[0]] == null) return;

  const bestAde = SCOREBOARD_MODELS.reduce((a, b) => (ade[b] < ade[a] ? b : a));
  const bestFde = SCOREBOARD_MODELS.reduce((a, b) => (fde[b] < fde[a] ? b : a));

  // Bars are 0-based (length ∝ actual km), so they never misrepresent the
  // magnitude. The models are genuinely close, so the small-but-real gap is
  // surfaced as an explicit "+% vs best" on the label rather than by distorting
  // the axis. A little headroom keeps the worst bar off the hard edge.
  const adeMax = Math.max(...SCOREBOARD_MODELS.map(mk => ade[mk])) * 1.05 || 1;
  const fdeMax = Math.max(...SCOREBOARD_MODELS.map(mk => fde[mk])) * 1.05 || 1;

  SCOREBOARD_MODELS.forEach(mk => {
    _setScoreBar(mk, ade[mk], fde[mk], adeMax, fdeMax,
                 ade[bestAde], fde[bestFde], mk === bestAde, mk === bestFde);
  });
}

// Label like "3.51 km +10%" — value plus how much worse than the best model.
function _valLabel(v, best, isBest) {
  if (v == null) return "—";
  if (isBest || !(best > 0)) return `${v.toFixed(2)} km`;
  const pct = Math.round((v - best) / best * 100);
  return `${v.toFixed(2)} km<span class="pp-delta">+${pct}%</span>`;
}

function _setScoreBar(mk, ade, fde, adeMax, fdeMax, adeBestV, fdeBestV, adeBest, fdeBest) {
  const adeFill = document.getElementById(`pp-ade-fill-${mk}`);
  const adeVal  = document.getElementById(`pp-ade-val-${mk}`);
  const fdeFill = document.getElementById(`pp-fde-fill-${mk}`);
  const fdeVal  = document.getElementById(`pp-fde-val-${mk}`);
  if (adeFill) adeFill.style.width = ade == null ? "0%" : `${Math.min(100, ade / adeMax * 100).toFixed(1)}%`;
  if (fdeFill) fdeFill.style.width = fde == null ? "0%" : `${Math.min(100, fde / fdeMax * 100).toFixed(1)}%`;
  if (adeVal) {
    adeVal.innerHTML = _valLabel(ade, adeBestV, adeBest);
    adeVal.classList.toggle("best", !!adeBest);
  }
  if (fdeVal) {
    fdeVal.innerHTML = _valLabel(fde, fdeBestV, fdeBest);
    fdeVal.classList.toggle("best", !!fdeBest);
  }
}

// ── Anomaly-detection metrics for the SELECTED model (REPLAY_MODEL): SET 1
//    (per class, optimal σ, injected only) + SET 2 (injected + real vs clean). ──
function renderDetectionMetrics() {
  const sec = document.getElementById("pp-detect");
  if (!sec) return;
  const bytype = (DETECTION_BY_TYPE && DETECTION_BY_TYPE[REPLAY_MODEL]) || {};
  if (!Object.keys(bytype).length) { sec.style.display = "none"; return; }
  sec.style.display = "";
  const pct = v => (v == null ? "—" : `${Math.round(v * 100)}%`);
  const set = (id, txt) => { const e = document.getElementById(id); if (e) e.textContent = txt; };
  set("pp-det-model", REPLAY_MODEL);

  // SET 1 — per-ship-type at each class's F1-optimal σ (injected only).
  const bt = document.getElementById("pp-det-bytype");
  if (bt) {
    const rows = Object.entries(bytype).sort((a, b) => b[1].f1 - a[1].f1);
    bt.innerHTML =
      `<div class="pp-bt-row pp-bt-hd"><span>Class</span><span>P</span><span>R</span><span>F1</span><span>σ</span></div>` +
      rows.map(([t, v]) => {
        const fc = v.f1 >= 0.5 ? "var(--green)" : v.f1 >= 0.35 ? "var(--accent-lt)" : "var(--muted)";
        return `<div class="pp-bt-row"><span>${t}</span>` +
          `<span>${pct(v.precision)}</span><span>${pct(v.recall)}</span>` +
          `<span style="color:${fc}">${v.f1.toFixed(2)}</span>` +
          `<span>${v.opt_sigma != null ? v.opt_sigma.toFixed(1) : "—"}</span></div>`;
      }).join("");
  }

  // SET 2 — combined: injected + real anomalies vs clean, per-type σ OR a rule.
  const c = DETECTION_COMBINED && DETECTION_COMBINED[REPLAY_MODEL];
  if (c) {
    set("pp-cf-n",         `${c.n_injected} injected + ${c.n_real} real · ${c.n_clean} clean`);
    set("pp-cf-precision", pct(c.precision));
    set("pp-cf-recall",    pct(c.recall));
    set("pp-cf-f1",        c.f1.toFixed(2));
    set("pp-cf-far",       pct(c.false_alarm_rate));
  }

  // Real physical-integrity faults (FLAGS): how many this model's detector surfaces.
  const rd = REAL_DETECTION && REAL_DETECTION[REPLAY_MODEL];
  const rr = document.getElementById("pp-det-real");
  if (rr) {
    if (rd && rd.n_real) {
      rr.style.display = "";
      set("pp-det-real-val", `${rd.flagged}/${rd.n_real} surfaced`);
    } else {
      rr.style.display = "none";
    }
  }
}

// ── Animation loop ────────────────────────────────────────────────────────────
function startLoop() {
  if (rafId) cancelAnimationFrame(rafId);
  lastTime = null;
  rafId    = requestAnimationFrame(animate);
}
function animate(ts) {
  if (!playing || simMode) { rafId = requestAnimationFrame(animate); return; }
  if (!lastTime) lastTime = ts;
  const tickMs = msPerTick / speedMult;
  if (ts - lastTime >= tickMs) {
    lastTime = ts;
    if (currentTick < WIN_LEN - 1) updateTick(currentTick + 1);
  }
  rafId = requestAnimationFrame(animate);
}

// ── Replay controls ────────────────────────────────────────────────────────────
function togglePlay() {
  if (replayEnded) { restartReplay(); return; }
  playing = !playing;
  const btn = document.getElementById("btn-play");
  btn.textContent = playing ? "⏸ Pause" : "▶ Play";
  btn.classList.toggle("active", playing);
}
function restartReplay() {
  replayEnded = false;
  updateTick(0);
  playing = true;
  const btn = document.getElementById("btn-play");
  btn.textContent = "⏸ Pause";
  btn.classList.add("active");
}
function resetReplay()  { restartReplay(); }
function seekTo(tick)   {
  if (tick < WIN_LEN - 1 && replayEnded) {
    replayEnded = false;
    const btn = document.getElementById("btn-play");
    btn.textContent = playing ? "⏸ Pause" : "▶ Play";
  }
  updateTick(tick);
}
function setSpeed(val)  {
  speedMult = +val;
  document.getElementById("speed-label").textContent = val + "×";
}
function toggleType(idx) {
  if (activeTypes.has(idx)) activeTypes.delete(idx); else activeTypes.add(idx);
  document.querySelectorAll(`[data-type="${idx}"]`).forEach(el =>
    el.classList.toggle("inactive", !activeTypes.has(idx)));
  if (simMode) _applyVisibilityFilter();
  else updateTick(currentTick);
}
// Replay-only: restrict the map to injected synthetic-anomaly vessels.
function toggleInjectedOnly() {
  injectedOnly = !injectedOnly;
  const el = document.getElementById("injected-filter");
  if (el) el.classList.toggle("active", injectedOnly);
  if (!simMode) updateTick(currentTick);
}
function toggleLevel(level) {
  if (activeLevels.has(level)) activeLevels.delete(level); else activeLevels.add(level);
  document.querySelectorAll(`.badge-${level}`).forEach(el =>
    el.classList.toggle("disabled", !activeLevels.has(level)));
  if (!simMode) updateTick(currentTick);
}
function jumpToShip(id) {
  const ship = ships[id];
  if (!ship) return;
  map.setView(ship.enc[SEQ_ENC - 1], 10, { animate: true });
  seekTo(SEQ_ENC);
  selectReplayShip(id);
}
function updateStats() {
  const n = ships.length;
  const anomalous = ships.filter(s => _shipLevel(s) !== "none").length;
  const ades = ships.map(s => s.ade_km);
  const meanAde = ades.length ? (ades.reduce((a, b) => a + b, 0) / ades.length).toFixed(2) : "—";
  document.getElementById("st-total").textContent = n;
  document.getElementById("st-anom").textContent  = anomalous;
  document.getElementById("st-ade").textContent   = meanAde + " km";
}

// ── Haversine (shared) ────────────────────────────────────────────────────────
function _haversineKm(a, b) {
  const R = 6371, toRad = d => d * Math.PI / 180;
  const dLat = toRad(b[0] - a[0]), dLon = toRad(b[1] - a[1]);
  const s = Math.sin(dLat/2)**2 + Math.cos(toRad(a[0])) * Math.cos(toRad(b[0])) * Math.sin(dLon/2)**2;
  return 2 * R * Math.asin(Math.sqrt(s));
}


// ══════════════════════════════════════════════════════════════════════════════
//  REPLAY SHIP SELECTION
// ══════════════════════════════════════════════════════════════════════════════

let selectedShipId    = null;
let replayStepMarkers = [];

function selectReplayShip(id) {
  if (simMode) return;
  if (selectedShipId === id) { deselectReplayShip(); return; }
  deselectReplayShip();
  selectedShipId = id;
  const ship = ships[id];
  const m    = markers[id];
  if (!ship || !m) return;
  m.marker.setIcon(_vesselIcon(TYPE_COLOR[ship.type] || "#64748b",
                               _tickCog(ship, currentTick), _shipLevel(ship), true, ship.type, ship.injected));
  _restyleReplayPreds(m, true);
  document.getElementById("sel-panel").classList.add("open");
  refreshReplaySelPanel(id);
}

function deselectReplayShip() {
  if (selectedShipId === null) return;
  const id = selectedShipId;
  selectedShipId = null;
  const ship = ships[id];
  const m    = markers[id];
  if (ship && m) {
    m.marker.setIcon(_vesselIcon(TYPE_COLOR[ship.type] || "#64748b",
                                 _tickCog(ship, currentTick), _shipLevel(ship), false, ship.type, ship.injected));
    _restyleReplayPreds(m, false);
  }
  replayStepMarkers.forEach(sm => map.removeLayer(sm));
  replayStepMarkers = [];
  document.getElementById("sel-panel").classList.remove("open");
}

// Re-apply every replay marker's icon — the severity ring depends on the selected
// model's per-class level, and updateTick() never re-sets icons. Called on model swap.
function _refreshMarkerIcons() {
  if (simMode || !markers) return;
  ships.forEach((ship, i) => {
    const m = markers[i];
    if (!m) return;
    m.marker.setIcon(_vesselIcon(TYPE_COLOR[ship.type] || "#64748b",
      _tickCog(ship, currentTick), _shipLevel(ship), selectedShipId === ship.id,
      ship.type, ship.injected));
  });
}

// Selection panel for replay: revealed predictions, drift between successive
// overlapping predictions over their shared steps, and error vs ground truth.
// Format a unix-seconds ping time as a compact UTC stamp (AIS times are UTC).
function _fmtTime(unix) {
  if (unix == null) return "—";
  const d = new Date(unix * 1000);
  const mon = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"][d.getUTCMonth()];
  const p = n => String(n).padStart(2, "0");
  return `${mon} ${p(d.getUTCDate())} ${p(d.getUTCHours())}:${p(d.getUTCMinutes())}:${p(d.getUTCSeconds())}Z`;
}

function refreshReplaySelPanel(id) {
  const ship = ships[id];
  if (!ship) return;
  const revealed = ship.predictions.filter(p => currentTick >= p.issued_step);
  const newest   = revealed[revealed.length - 1] || null;

  document.getElementById("sel-mmsi").textContent =
    ship.mmsi ? `MMSI ${ship.mmsi}` : `Ship #${ship.id}`;
  document.getElementById("sel-type").textContent =
    `${ship.type_name} · replay · ${_shipLevel(ship)}`;

  const injRow = ship.injected
    ? `<div class="sel-row"><span>&#128137; Injected</span><span class="sel-val" style="color:#f3ca86">${ship.anomaly_type.replace(/_/g," ")} · ${ship.severity} ${ship.severity_unit}</span></div>`
    : "";

  // Alerts triggered on this vessel — which kind, at what step, and when (ping time).
  const alerts = _vesselAlerts(ship).slice().sort((a, b) => (a.tick ?? 0) - (b.tick ?? 0));
  let alertHtml = `<div class="sel-alert-title">Alerts triggered</div>`;
  if (alerts.length) {
    alertHtml += alerts.map(a => {
      const m = REPLAY_KINDS[a.kind] || {};
      const t = (ship.times && a.tick != null) ? ship.times[a.tick] : null;
      return `<div class="sel-alert-row" style="--c:${m.col || '#8494a0'}">
        <div class="sel-alert-hd"><span class="sel-alert-dot"></span>
          <span class="sel-alert-lbl">${m.label || a.kind}</span>
          <span class="sel-alert-step">step ${a.tick}</span></div>
        <div class="sel-alert-meta">${_fmtTime(t)}${a.reason ? " · " + a.reason : ""}</div>
      </div>`;
    }).join("");
  } else {
    alertHtml += `<div class="sel-alert-none">No alerts on this vessel</div>`;
  }

  const pingRow = (ship.times && ship.times[currentTick] != null)
    ? `<div class="sel-row"><span>Ping time</span><span class="sel-val">${_fmtTime(ship.times[currentTick])}</span></div>`
    : "";

  let html = injRow + pingRow + `
    <div class="sel-row"><span>Mean ADE</span><span class="sel-val">${ship.ade_km.toFixed(2)} km</span></div>
    <div class="sel-row"><span>Predictions issued</span><span class="sel-val">${revealed.length} / ${ship.predictions.length}</span></div>`
    + alertHtml;

  if (!newest) {
    html += `<div class="sel-watch">History phase — first prediction at step ${ship.predictions[0].issued_step}</div>`;
  } else {
    // Error vs ground truth per revealed prediction (mean over its 10 steps),
    // for the currently-selected model.
    html += `<div class="sel-pred-title">${REPLAY_MODEL} vs ground truth</div>`;
    revealed.slice(-4).reverse().forEach(p => {
      const ade = (p.model_ade && p.model_ade[REPLAY_MODEL] != null) ? p.model_ade[REPLAY_MODEL] : p.ade_km;
      const col = ade > 3 ? "#ef4444" : ade > 1 ? "#f97316" : "#22c55e";
      html += `<div class="sel-cmp-row">
        <span class="sel-cmp-age">#${p.pred_id + 1} @ step ${p.issued_step}</span>
        <span class="sel-cmp-drift" style="color:${col}">${ade.toFixed(2)} km</span></div>`;
    });

    // Drift between each older revealed prediction and the newest, computed
    // over the absolute steps both predictions cover.
    if (revealed.length > 1) {
      html += `<div class="sel-cmp-title">Drift vs latest (shared steps)</div>`;
      const nc = _predCoords(newest);
      for (let i = revealed.length - 2; i >= Math.max(0, revealed.length - 4); i--) {
        const older = revealed[i];
        const oc = _predCoords(older);
        let sum = 0, k = 0;
        for (let st = newest.issued_step;
             st < older.issued_step + oc.length &&
             st < newest.issued_step + nc.length; st++) {
          sum += _haversineKm(oc[st - older.issued_step], nc[st - newest.issued_step]);
          k++;
        }
        const drift = k ? sum / k : 0;
        const col = drift > 3 ? "#ef4444" : drift > 1 ? "#f97316" : "#22c55e";
        html += `<div class="sel-cmp-row">
          <span class="sel-cmp-age">#${older.pred_id + 1} vs #${newest.pred_id + 1} · ${k} steps</span>
          <span class="sel-cmp-drift" style="color:${col}">${drift.toFixed(2)} km</span></div>`;
      }
    }
  }
  document.getElementById("sel-body").innerHTML = html;

  // Step markers on the newest revealed prediction
  replayStepMarkers.forEach(sm => map.removeLayer(sm));
  replayStepMarkers = [];
  if (newest) {
    const col = TYPE_COLOR[ship.type] || "#64748b";
    _predCoords(newest).forEach((pos, i) => {
      const sm = L.circleMarker(pos, {
        radius: 4, fillColor: "#111", fillOpacity: 0.85, color: col, weight: 2,
      }).bindTooltip(`<b>Step ${newest.issued_step + i + 1}</b><br>` +
                     `${pos[0].toFixed(4)}°N ${pos[1].toFixed(4)}°E<br>` +
                     `<span style="opacity:.75">${_fmtTime(ship.times && ship.times[newest.issued_step + i])}</span>`,
                     { direction: "top" }).addTo(map);
      replayStepMarkers.push(sm);
    });
  }
}


// ══════════════════════════════════════════════════════════════════════════════
//  MODE SWITCHING  (Replay ⇄ Live Sim)
// ══════════════════════════════════════════════════════════════════════════════

let simMode = false;

function setMode(mode) {
  if (mode === "sim"    && !simMode) { _enterSimMode(); _ensureSSE(); fetchSimStatus(); }
  if (mode === "replay" &&  simMode) {
    // Pause the engine so it doesn't burn CPU while nobody is watching,
    // then return to the static replay view. Sim state survives — switching
    // back and pressing Resume continues where it left off.
    fetch("/api/sim/pause", { method: "POST" }).catch(() => {});
    if (evtSource) { evtSource.close(); evtSource = null; }
    _exitSimMode();
  }
}

function _setModeButtons() {
  document.getElementById("mode-replay").classList.toggle("active", !simMode);
  document.getElementById("mode-sim").classList.toggle("active", simMode);
}

function _enterSimMode() {
  simMode = true; playing = false;
  deselectReplayShip();
  markers.forEach(m => {
    if (map.hasLayer(m.marker)) map.removeLayer(m.marker);
    m.trail.setLatLngs([]); m.gtLine.setLatLngs([]);
    m.predLines.forEach(pl => { if (pl.shown) { map.removeLayer(pl.line); pl.shown = false; } });
  });
  _setModeButtons();
  buildAnomFilter();
  document.getElementById("sim-bar").classList.remove("hidden");
  document.getElementById("replay-controls").style.display    = "none";
  document.getElementById("stats-title").textContent          = "LIVE SIM";
  document.getElementById("st-l-anom").textContent            = "Anomalous ships";
  document.getElementById("st-l-ade").textContent             = "Anomalous events";
  document.getElementById("mov-filter-section").style.display = "";
  document.getElementById("anom-level-section").style.display = "none";
  // Injection is a Replay-only concept — hide the filter and clear it in Sim.
  injectedOnly = false;
  const _inj = document.getElementById("injected-filter");
  if (_inj) { _inj.style.display = "none"; _inj.classList.remove("active"); }
  const phase = document.getElementById("phase-badge");
  phase.textContent = "⬤ LIVE" + _modelTag(); phase.className = "sim";
  _setAnomPanel(_anomOpen);            // move the panel state onto the sim feed
}

function _exitSimMode() {
  simMode = false; simPaused = false;
  _clearSimVessels();
  _clearAlertMarker();
  _setModeButtons();
  _setAnomPanel(_anomOpen);            // move the panel state onto the replay list
  document.getElementById("sim-bar").classList.add("hidden");
  document.getElementById("sel-panel").classList.remove("open");
  document.getElementById("replay-controls").style.display    = "";
  document.getElementById("stats-title").textContent          = "Fleet";
  document.getElementById("st-l-anom").textContent            = "Anomalous";
  document.getElementById("st-l-ade").textContent             = "Mean ADE";
  document.getElementById("mov-filter-section").style.display = "none";
  document.getElementById("anom-level-section").style.display = "";
  { const _inj = document.getElementById("injected-filter");
    if (_inj && ships.some(s => s.injected)) _inj.style.display = ""; }
  createMarkers();
  restartReplay();
  startLoop();
  _setActiveModel(REPLAY_MODEL);        // reflect the Replay model selection
}


// ══════════════════════════════════════════════════════════════════════════════
//  LIVE SIM MODE
// ══════════════════════════════════════════════════════════════════════════════

let simPaused      = false;
let evtSource      = null;
const simVessels   = new Map();
const SIM_TRAIL_LEN  = 15;
let SIM_SPEED_VALS   = [60, 300, 900, 1728, 3600, 7200, 14400];
const MAX_PREDS_DRAWN = 4;

let dataStartTs = null;
let dataEndTs   = null;
let scrubbing   = false;
let seeking     = false;

// Prediction-rate bookkeeping (client-side, wall-clock)
const _predArrivals = [];
let _lastPredSimTime = null;

function _notePredArrival(simTime) {
  const now = performance.now();
  _predArrivals.push(now);
  while (_predArrivals.length && now - _predArrivals[0] > 10000) _predArrivals.shift();
  _lastPredSimTime = simTime;
  _updatePredReadout();
}
function _updatePredReadout() {
  const now  = performance.now();
  while (_predArrivals.length && now - _predArrivals[0] > 10000) _predArrivals.shift();
  const rate = _predArrivals.length / 10;
  const rateEl = document.getElementById("sim-predrate-disp");
  const lastEl = document.getElementById("sim-lastpred-disp");
  if (rateEl) rateEl.textContent = rate >= 0.05 ? rate.toFixed(1) + "/s" : "—";
  if (lastEl) lastEl.textContent = _lastPredSimTime || "—";
}
setInterval(() => { if (simMode) _updatePredReadout(); }, 1000);

// Movement filter
let movFilter = "all";
function toggleMovFilter(filter) {
  movFilter = filter;
  ["all", "underway", "anchor"].forEach(f => {
    const btn = document.getElementById(`mf-${f}`);
    if (btn) btn.classList.toggle("active", f === filter);
  });
  _applyVisibilityFilter();
}
function _vesselVisible(v) {
  if (!activeTypes.has(v.ship_type)) return false;
  if (movFilter === "underway" && v.lastSog < 1.0) return false;
  if (movFilter === "anchor"   && v.lastSog >= 1.0) return false;
  return true;
}
function _applyVisibilityFilter() {
  simVessels.forEach(v => {
    const vis = _vesselVisible(v);
    if (vis === v._vis) return;
    v._vis = vis;
    if (vis) {
      if (!map.hasLayer(v.marker))    v.marker.addTo(map);
      if (!map.hasLayer(v.trailLine)) v.trailLine.addTo(map);
      v.predLines.forEach(pl => { if (!map.hasLayer(pl.line)) pl.line.addTo(map); });
    } else {
      map.removeLayer(v.marker);
      map.removeLayer(v.trailLine);
      v.predLines.forEach(pl => map.removeLayer(pl.line));
    }
  });
}

// Batched icon update — called once per SSE batch, not per ping
const _dirtyVessels = new Set();
function _scheduleIconUpdate(mmsi) { _dirtyVessels.add(mmsi); }
function _flushDirtyIcons() {
  if (!_dirtyVessels.size) return;
  _dirtyVessels.forEach(mmsi => {
    const v = simVessels.get(mmsi);
    if (!v) return;
    const isSel    = selectedMmsi === mmsi;
    const anomLevel = v.lastAnomaly === null ? "none" :
                      v.lastAnomaly > 4.5 ? "severe"   :
                      v.lastAnomaly > 3.0 ? "moderate" :
                      v.lastAnomaly > 1.5 ? "mild"     : "none";
    v.marker.setIcon(_vesselIcon(v.color, v.lastCog, anomLevel, isSel, v.ship_type));
    // Vessels still warming up (< SEQ_ENC pings) render translucent so the
    // "not predicting yet" state is visible at a glance.
    v.marker.setOpacity((v.holding || v.nPings < SEQ_ENC) ? 0.55 : 1.0);
    if (v.inferPending) {
      const el = v.marker.getElement();
      if (el) el.classList.add("infer-pending");
    }

    const vis = _vesselVisible(v);
    if (vis !== v._vis) {
      v._vis = vis;
      if (vis) {
        if (!map.hasLayer(v.marker))    v.marker.addTo(map);
        if (!map.hasLayer(v.trailLine)) v.trailLine.addTo(map);
        v.predLines.forEach(pl => { if (!map.hasLayer(pl.line)) pl.line.addTo(map); });
      } else {
        map.removeLayer(v.marker);
        map.removeLayer(v.trailLine);
        v.predLines.forEach(pl => map.removeLayer(pl.line));
      }
    }
  });
  _dirtyVessels.clear();
}

// ── Selection state ────────────────────────────────────────────────────────────
let selectedMmsi    = null;
let selStepMarkers  = [];
let selOriginMarker = null;
let selHistLine     = null;

function fetchSimStatus() {
  fetch("/api/sim/status").then(r => r.json()).then(applySimStatus).catch(() => {});
}

function applySimStatus(s) {
  if (!s) return;
  if (s.sim_time && !seeking)
    document.getElementById("sim-time-disp").textContent = s.sim_time;
  if (s.pings_processed !== undefined)
    document.getElementById("sim-pings-disp").textContent = s.pings_processed.toLocaleString();
  if (s.active_vessels !== undefined)
    document.getElementById("sim-vessels-disp").textContent = s.active_vessels;
  if (s.anomalies_flagged !== undefined)
    document.getElementById("sim-anom-disp").textContent = s.anomalies_flagged;
  if (s.predictions_made !== undefined)
    document.getElementById("sim-pred-disp").textContent = s.predictions_made;

  if (Array.isArray(s.speed_steps) && s.speed_steps.length) SIM_SPEED_VALS = s.speed_steps;

  if (s.data_start_ts) dataStartTs = s.data_start_ts;
  if (s.data_end_ts)   dataEndTs   = s.data_end_ts;
  if (s.data_start) document.getElementById("sim-data-start").textContent = s.data_start;
  if (s.data_end)   document.getElementById("sim-data-end").textContent   = s.data_end;

  if (s.predict_every) {
    document.getElementById("sim-predevery-slider").value = s.predict_every;
    document.getElementById("sim-predevery-label").textContent = s.predict_every + " pings";
  }

  const dot = document.getElementById("sim-state-dot");
  const colors = { running: "#59b39a", paused: "#d0803f", seeking: "#e0a24a",
                   stopped: "#64748b", done: "#64748b" };
  dot.style.background = colors[s.state] || "#64748b";

  const isRunning = s.state === "running";
  const isPaused  = s.state === "paused";
  const isLive    = isRunning || isPaused;   // a session exists
  document.getElementById("sim-btn-start").classList.toggle("active", isRunning);
  document.getElementById("sim-btn-pause").classList.toggle("active", isPaused);
  document.getElementById("sim-btn-stop").classList.toggle("active", isLive);
  document.getElementById("sim-btn-pause").textContent = isPaused ? "▶ Resume" : "⏸ Pause";
  simPaused = isPaused;
}

// ── Model selection ──────────────────────────────────────────────────────────
// Each card in the Model panel is a button: clicking it hot-swaps the model the
// sim runs inference with. The server clears live predictions on swap, so we
// drop the drawn ones too (see the "reset_preds" SSE event).
let _modelSwitching = false;

function _setActiveModel(key) {
  ACTIVE_MODEL = key;
  document.querySelectorAll(".pp-card").forEach(c => {
    const on = c.dataset.model === key;
    c.classList.toggle("pp-active", on);
    c.classList.remove("pp-failed");
    const badge = c.querySelector(".act-badge");
    if (badge) badge.hidden = !on;
  });
  // Refresh the badge so a model switch shows immediately even while paused
  // (Replay's per-tick rewrite won't fire when the clock is stopped).
  const phase = document.getElementById("phase-badge");
  if (phase) {
    if (simMode) {
      phase.textContent = "⬤ LIVE" + _modelTag();
    } else {
      const inDecoder = currentTick >= SEQ_ENC;
      const atEnd     = currentTick >= WIN_LEN - 1;
      phase.textContent = (atEnd ? "⬤ COMPLETE" : inDecoder ? "⬤ PREDICTING" : "⬤ HISTORY") + _modelTag();
    }
  }
}

// Redraw every shown Replay prediction line + selected-panel with REPLAY_MODEL.
function _redrawReplayPreds() {
  markers.forEach((m, id) => {
    const ship = ships[id];
    m.predLines.forEach(pl => {
      if (!pl.shown) return;
      const originIdx = pl.pred.issued_step - SEQ_ENC;
      const origin = originIdx <= 0 ? ship.enc[SEQ_ENC - 1] : ship.future[originIdx - 1];
      pl.line.setLatLngs([origin, ..._predCoords(pl.pred)]);
    });
  });
  if (selectedShipId != null) refreshReplaySelPanel(selectedShipId);
}

function selectModel(key) {
  const card = document.querySelector(`.pp-card[data-model="${key}"]`);
  if (!card || card.classList.contains("pp-active")) return;   // already selected

  // Replay: swap which precomputed model path is drawn + which model's detection
  // metrics are shown (no live inference).
  if (!simMode) {
    REPLAY_MODEL = key;
    _setActiveModel(key);
    _redrawReplayPreds();
    renderDetectionMetrics();
    restartReplay();         // replay the run from the start under the newly-picked model
    buildReplayFilter();     // feed-kind counts are model-dependent
    _refreshMarkerIcons();   // recolour severity rings for the newly-picked model
    return;
  }

  if (_modelSwitching) return;
  _modelSwitching = true;
  document.querySelector(".pp-cards").classList.add("pp-loading");

  fetch("/api/sim/model", {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ model: key }),
  })
    .then(r => r.json().then(d => ({ ok: r.ok, d })))
    .then(({ ok, d }) => {
      if (!ok) throw new Error(d.error || "switch failed");
      _setActiveModel(d.model || key);
      _clearAllPredLines();          // old model's tracks are no longer valid
    })
    .catch(err => {
      console.error("model switch failed:", err);
      card.classList.add("pp-failed");
    })
    .finally(() => {
      _modelSwitching = false;
      document.querySelector(".pp-cards").classList.remove("pp-loading");
    });
}

// Keyboard support for the card-buttons.
document.addEventListener("keydown", e => {
  const el = e.target;
  if ((e.key === "Enter" || e.key === " ") && el && el.classList
      && el.classList.contains("pp-card")) {
    e.preventDefault();
    selectModel(el.dataset.model);
  }
});

// Drop every drawn prediction (keeps vessels, trails and history).
function _clearAllPredLines() {
  simVessels.forEach(v => {
    v.predLines.forEach(pl => map.removeLayer(pl.line));
    v.predLines = [];
    v.inferPending = false;
    const el = v.marker.getElement();
    if (el) el.classList.remove("infer-pending");
  });
  _clearSelStepMarkers();
  if (selectedMmsi !== null) refreshSelPanel(selectedMmsi);
}

// ── Model Performance Panel ──────────────────────────────────────────────────
let _perfOpen   = false;
let _perfPollId = null;

function togglePerfPanel() {
  _perfOpen = !_perfOpen;
  document.getElementById("perf-panel").classList.toggle("open", _perfOpen);
  document.getElementById("btn-perf").classList.toggle("active", _perfOpen);
  if (_perfOpen) {
    _updatePerfLive();
    _perfPollId = setInterval(_updatePerfLive, 2000);
  } else {
    clearInterval(_perfPollId);
    _perfPollId = null;
  }
}

function _updatePerfLive() {
  fetch("/api/sim/status").then(r => r.json()).then(s => {
    // Only sync the active card to the sim's model in Live Sim — in Replay the
    // active card reflects the user's REPLAY_MODEL choice, not the sim engine.
    if (simMode && s.model && !_modelSwitching) _setActiveModel(s.model);
    const thresh = (s.anomaly_threshold ?? 3.0).toFixed(1);
    document.getElementById("pp-thresh").textContent = thresh + " σ";
    document.getElementById("pp-every").textContent  = (s.predict_every ?? 5) + " pings";
    document.getElementById("pp-maxv").textContent   = s.max_vessels ?? 50;
    document.getElementById("pp-speed").textContent = s.speed ? s.speed + "×" : "—";

    const pings = s.pings_processed || 0;
    const preds = s.predictions_made || 0;
    const dets  = s.anomalies_flagged || 0;
    document.getElementById("pp-pings").textContent = pings > 0 ? pings.toLocaleString() : "—";
    document.getElementById("pp-preds").textContent = preds > 0 ? preds.toLocaleString() : "—";
    document.getElementById("pp-dets").textContent  = dets  > 0 ? dets.toLocaleString()  : "—";
    document.getElementById("pp-rate").textContent  = preds > 0 ? (dets / preds * 100).toFixed(1) + "%" : "—";
  }).catch(() => {});
}

// ── Sim config controls ─────────────────────────────────────────────────────────
function _postConfig(body) {
  fetch("/api/sim/configure", {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  }).catch(() => {});
}
function onSimSpeedChange(sliderVal) {
  const speed = SIM_SPEED_VALS[+sliderVal] || 900;
  document.getElementById("sim-speed-label").textContent = speed + "×";
  _postConfig({ speed });
}
function onSimThreshChange(val) {
  document.getElementById("sim-thresh-label").textContent = val + " σ";
  _postConfig({ anomaly_threshold: +val });
}
function onSimPredictEveryChange(val) {
  document.getElementById("sim-predevery-label").textContent = val + " pings";
  _postConfig({ predict_every: +val });
}
function onSimMaxVesselsChange(val) {
  _postConfig({ max_vessels: +val });
}

// ── Timeline scrubber ───────────────────────────────────────────────────────────
function onSimProgressInput(val) {
  scrubbing = true;
  const frac = +val / 1000;
  document.getElementById("sim-progress-pct").textContent = (frac * 100).toFixed(1) + "%";
  if (dataStartTs && dataEndTs) {
    const ts = dataStartTs + frac * (dataEndTs - dataStartTs);
    const d  = new Date(ts * 1000);
    document.getElementById("sim-time-disp").textContent =
      d.toISOString().slice(0, 19).replace("T", " ");
  }
}
function onSimProgressCommit(val) {
  const frac = +val / 1000;
  if (!simMode) setMode("sim");
  _ensureSSE();
  _setSeeking(true);
  fetch("/api/sim/seek", {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ fraction: frac }),
  }).then(r => r.json()).then(applySimStatus).catch(() => { _setSeeking(false); });
  setTimeout(() => { scrubbing = false; }, 300);
}

// Loading feedback while a seek is in flight: the server broadcasts
// state:"seeking" immediately, then a reset+status(seeked) batch when the
// stream has reopened at the target time.
function _setSeeking(on) {
  seeking = on;
  const lbl = document.getElementById("sim-progress-pct");
  if (on) lbl.textContent = "Seeking…";
  document.getElementById("sim-progress").disabled = on;
  document.getElementById("sim-bar").classList.toggle("seeking", on);
}

// ── SSE connection ────────────────────────────────────────────────────────────
function _ensureSSE() {
  if (evtSource) return;
  evtSource = new EventSource("/api/sim/events");
  evtSource.onmessage = e => {
    try {
      const batch = JSON.parse(e.data);
      if (Array.isArray(batch)) {
        batch.forEach(handleSimEvent);
        _flushDirtyIcons();
      }
    } catch (_) {}
  };
  evtSource.onerror = () => {};
}

// ── Sim lifecycle ─────────────────────────────────────────────────────────────
function _clearSimVessels() {
  deselectSimVessel();
  clearVesselFocus();
  simVessels.forEach(v => {
    map.removeLayer(v.marker);
    map.removeLayer(v.trailLine);
    v.predLines.forEach(pl => map.removeLayer(pl.line));
  });
  simVessels.clear();
  _dirtyVessels.clear();
}

function simStart() {
  if (!simMode) setMode("sim");
  _clearSimVessels();

  const btn = document.getElementById("sim-btn-start");
  btn.textContent = "⏳ Starting…"; btn.disabled = true;

  if (evtSource) { evtSource.close(); evtSource = null; }
  _ensureSSE();

  fetch("/api/sim/start", { method: "POST" })
    .then(r => { if (!r.ok) throw new Error(r.status); return r.json(); })
    .then(applySimStatus)
    .catch(err => { console.error("sim start failed:", err); btn.textContent = "⚠ Error"; })
    .finally(() => { btn.textContent = "▶ Start"; btn.disabled = false; });
}

function simTogglePause() {
  const ep = simPaused ? "/api/sim/resume" : "/api/sim/pause";
  fetch(ep, { method: "POST" }).then(r => r.json()).then(applySimStatus).catch(() => {});
}

function simStop() {
  fetch("/api/sim/stop", { method: "POST" }).then(r => r.json()).then(applySimStatus).catch(() => {});
}

function simReset() {
  fetch("/api/sim/reset", { method: "POST" }).catch(() => {});
  if (evtSource) { evtSource.close(); evtSource = null; }
  simPaused = false;
  document.getElementById("sim-feed-entries").innerHTML = "";
  _feedByMmsi.clear();
  _clearAlertMarker();
  Object.keys(_kindTotals).forEach(k => delete _kindTotals[k]);
  document.querySelectorAll(".ac-n").forEach(el => el.textContent = "0");
  const ft = document.getElementById("feed-total"); if (ft) ft.textContent = "";
  document.getElementById("sim-progress").value = 0;
  document.getElementById("sim-progress-pct").textContent = "0.0%";
  _setSeeking(false);
  _exitSimMode();  // clears sim vessels, restores static replay
}

// ── SSE event handlers ────────────────────────────────────────────────────────
function handleSimEvent(ev) {
  switch (ev.type) {
    case "ping":          onSimPing(ev);         break;
    case "prediction":    onSimPrediction(ev);   break;
    case "infer_pending": onSimInferPending(ev); break;
    case "anomaly":       onSimAnomaly(ev);      break;
    case "status":        onSimStatus(ev);       break;
    case "reset":         _clearSimVessels();    break;
    case "reset_preds":   _clearAllPredLines();  break;   // model was swapped
  }
}

function _makeVessel(mmsi, lat, lon, cog, ship_type) {
  const color = TYPE_COLOR[ship_type] ?? TYPE_COLOR[0];
  const icon  = _vesselIcon(color, cog, "none", false, ship_type);
  const marker = L.marker([lat, lon], { icon, zIndexOffset: 100 }).addTo(map);
  const trailLine = L.polyline([], { color, weight: 1.5, opacity: 0.4 }).addTo(map);
  marker.bindTooltip("", { sticky: true, opacity: 0.92, className: "" });
  marker.on("click", e => { L.DomEvent.stopPropagation(e); selectSimVessel(mmsi); });
  const v = {
    marker, trailLine,
    predLines: [],
    trail: [], history: [],
    ship_type, color,
    lastSog: 0, lastCog: cog, lastAnomaly: null, nPings: 0,
    inferPending: false, lastPredAt: null, holding: false,
    _vis: true,
  };
  simVessels.set(mmsi, v);
  return v;
}

function onSimPing(ev) {
  const { mmsi, lat, lon, sog, cog, ship_type, n_pings, anomaly_score, flagged, holding } = ev;
  let v = simVessels.get(mmsi) || _makeVessel(mmsi, lat, lon, cog, ship_type);

  // A vessel's type usually arrives a few pings after its first position
  // (AIS SHIP_TYPE=0 = "not available"), so adopt it whenever it changes —
  // otherwise the icon stays locked on the grey "Unknown" circle.
  if (ship_type !== v.ship_type) {
    v.ship_type = ship_type;
    v.color = TYPE_COLOR[ship_type] ?? TYPE_COLOR[0];
    v.trailLine.setStyle({ color: v.color });
    v.predLines.forEach(pl => pl.line.setStyle({ color: v.color }));
    if (selHistLine && selectedMmsi === mmsi) selHistLine.setStyle({ color: v.color });
    _scheduleIconUpdate(mmsi);   // rebuild the icon: new colour and shape
  }

  v.trail.push([lat, lon]);
  if (v.trail.length > SIM_TRAIL_LEN) v.trail.shift();
  v.history.push([lat, lon]);
  v.lastSog = sog; v.lastCog = cog; v.lastAnomaly = anomaly_score; v.nPings = n_pings;

  // Stationary vessel: the model is skipped ("holding position"). Drop any
  // predicted tracks it had while moving so a parked ship shows no forecast.
  if (holding && !v.holding && v.predLines.length) {
    v.predLines.forEach(pl => map.removeLayer(pl.line));
    v.predLines = [];
    if (selectedMmsi === mmsi) _clearSelStepMarkers();
  }
  v.holding = holding;

  v.marker.setLatLng([lat, lon]);
  v.trailLine.setLatLngs(v.trail);
  if (_focusMmsi === mmsi && _focusRing) _focusRing.setLatLng([lat, lon]);
  _scheduleIconUpdate(mmsi);   // _flushDirtyIcons dims holding/warming-up vessels

  const zText = anomaly_score !== null
    ? `<br>z: <b style="color:${flagged ? "#ef4444" : "#f59e0b"}">${anomaly_score.toFixed(2)}</b>` : "";
  const stateText = holding
    ? `<br><small style="color:#94a3b8">⚓ holding position (stationary)</small>`
    : n_pings < SEQ_ENC
      ? `<br><small style="color:#e0a24a">Warming up… ${n_pings}/${SEQ_ENC} pings</small>`
      : v.inferPending
        ? `<br><small style="color:#e0a24a">⚡ predicting…</small>` : "";
  v.marker.setTooltipContent(
    `<b>${mmsi}</b><br>${GROUP_NAMES[ship_type]}<br>SOG ${sog.toFixed(1)} kn  COG ${cog.toFixed(0)}°${zText}${stateText}`);

  if (selectedMmsi === mmsi) {
    if (selHistLine) selHistLine.setLatLngs(v.history);
    refreshSelPanel(mmsi);
  }
}

function onSimInferPending(ev) {
  const v = simVessels.get(ev.mmsi);
  if (!v) return;
  v.inferPending = true;
  const el = v.marker.getElement();
  if (el) el.classList.add("infer-pending");
  // Safety: clear after 4s if the result never lands (error path server-side)
  setTimeout(() => {
    if (v.inferPending) {
      v.inferPending = false;
      const el2 = v.marker.getElement();
      if (el2) el2.classList.remove("infer-pending");
    }
  }, 4000);
  if (selectedMmsi === ev.mmsi) refreshSelPanel(ev.mmsi);
}

function onSimPrediction(ev) {
  const { mmsi, pred, sogs, pred_id, issued_ts } = ev;
  const v = simVessels.get(mmsi);
  if (!v || !pred.length) return;

  v.inferPending = false;
  v.lastPredAt   = issued_ts;
  const mel = v.marker.getElement();
  if (mel) mel.classList.remove("infer-pending");

  const origin = v.trail[v.trail.length - 1] || pred[0];
  // New prediction arrives bright and flashing, then settles into the
  // age-faded stack via _restylePredLines.
  const line = L.polyline([origin, ...pred], {
    color: v.color, weight: 3.5, opacity: 1.0, dashArray: "6 5",
  }).addTo(map);
  flashLine(line);

  v.predLines.push({ pred_id, issued: issued_ts, line, coords: pred, sogs: sogs || [] });

  while (v.predLines.length > MAX_PREDS_DRAWN) {
    const old = v.predLines.shift();
    map.removeLayer(old.line);
  }
  setTimeout(() => _restylePredLines(mmsi), 900);
  _notePredArrival(issued_ts);

  if (selectedMmsi === mmsi) { _drawSelStepMarkers(mmsi); refreshSelPanel(mmsi); }
}

function _restylePredLines(mmsi) {
  const v = simVessels.get(mmsi);
  if (!v) return;
  const isSel = selectedMmsi === mmsi;
  const n = v.predLines.length;
  v.predLines.forEach((pl, i) => {
    const age    = n - 1 - i;
    const newest = age === 0;
    const baseOp = newest ? (isSel ? 1.0 : 0.6) : Math.max(0.1, (isSel ? 0.5 : 0.3) - age * 0.1);
    const w      = newest ? (isSel ? 3.5 : 2)   : (isSel ? 2 : 1.2);
    pl.line.setStyle({ opacity: baseOp, weight: w });
  });
}

// Per anomaly kind: a unique colour and a short label for the feed.
const ANOM_KINDS = {
  speed_jump:      { col: "#e0554e", label: "Position jump" },
  on_land:         { col: "#d98a4e", label: "On land" },
  impossible_sog:  { col: "#d9b24e", label: "Impossible speed" },
  loitering:       { col: "#8ea0ac", label: "Loitering",
                     note: "Cargo, tanker & passenger only — excludes vessels docked in port or at a berth" },
  model_deviation: { col: "#5f95b5", label: "Off predicted path" },
};

// Readable text colour for a given chip background (dark ink on light, light on dark).
function _txtOn(hex) {
  const r = parseInt(hex.slice(1, 3), 16),
        g = parseInt(hex.slice(3, 5), 16),
        b = parseInt(hex.slice(5, 7), 16);
  return (0.299 * r + 0.587 * g + 0.114 * b) > 140 ? "#04140a" : "#e8f2f4";
}
Object.values(ANOM_KINDS).forEach(m => { m.txt = _txtOn(m.col); });

// ── Alert focus ────────────────────────────────────────────────────────────
// Clicking an alert pans to the vessel and rings it in yellow. The ring tracks
// the vessel as it moves (see onSimPing) so it stays easy to pick out.
let _focusMmsi = null;
let _focusRing = null;

function focusVessel(mmsi) {
  // Clicking the already-focused alert toggles the ring back off. Checked
  // before the vessel lookup so you can always un-focus, even if the vessel
  // has since dropped out of the working set.
  if (_focusMmsi === mmsi) { clearVesselFocus(); return; }

  const v = simVessels.get(mmsi);
  if (!v) return;
  _focusMmsi = mmsi;

  const pos = v.marker.getLatLng();
  if (!_focusRing) {
    _focusRing = L.circleMarker(pos, {
      radius: 15, color: "#facc15", weight: 3, fill: false,
      interactive: false, className: "focus-ring",
    }).addTo(map);
  } else {
    _focusRing.setLatLng(pos);
    if (!map.hasLayer(_focusRing)) _focusRing.addTo(map);
  }
  _focusRing.bringToFront();
  map.setView(pos, Math.max(map.getZoom(), 11), { animate: true });

  // Mark which alert box the focus belongs to.
  document.querySelectorAll(".feed-entry.focused")
          .forEach(el => el.classList.remove("focused"));
  const entry = _feedByMmsi.get(mmsi);
  if (entry) entry.el.classList.add("focused");
}

function clearVesselFocus() {
  _focusMmsi = null;
  if (_focusRing && map.hasLayer(_focusRing)) map.removeLayer(_focusRing);
  document.querySelectorAll(".feed-entry.focused")
          .forEach(el => el.classList.remove("focused"));
}

// One feed box per ship: repeated alerts update the box in place, bump a
// per-kind counter, and move it to the top instead of flooding the feed.
const _feedByMmsi = new Map();   // mmsi -> { el, kinds: {kind: n}, instances: {kind: {...}} }
let _alertMarker = null;         // pulse marker for the currently-shown alert instance
let _alertMarkerKey = null;      // "mmsi:kind" currently shown, for click-to-toggle

// ── Alert-type filter ──────────────────────────────────────────────────────
const _activeAnomKinds = new Set(Object.keys(ANOM_KINDS));
const _kindTotals = {};   // kind -> total alerts across all ships (for chip counts)

function buildAnomFilter() {
  const wrap = document.getElementById("anom-filter");
  if (!wrap || wrap.childElementCount) return;   // build once
  wrap.innerHTML = Object.entries(ANOM_KINDS).map(([k, m]) =>
    `<span class="anom-chip active" data-kind="${k}" style="--c:${m.col};--tc:${m.txt}"
       title="${m.note || m.label}" onclick="toggleAnomKind('${k}')">
       <span class="chip-dot"></span><span>${m.label}</span>
       <span class="ac-n" data-count="${k}">0</span></span>`).join("");
}

function toggleAnomKind(kind) {
  if (_activeAnomKinds.has(kind)) _activeAnomKinds.delete(kind);
  else _activeAnomKinds.add(kind);
  document.querySelectorAll(`.anom-chip[data-kind="${kind}"]`).forEach(el =>
    el.classList.toggle("active", _activeAnomKinds.has(kind)));
  _feedByMmsi.forEach(_applyFeedVisibility);
}

// A ship's box is shown if it carries at least one still-enabled alert kind.
function _applyFeedVisibility(entry) {
  const anyActive = Object.keys(entry.kinds).some(k => _activeAnomKinds.has(k));
  entry.el.style.display = anyActive ? "" : "none";
}

function onSimAnomaly(ev) {
  const { mmsi, ship_type, sim_time, sog, step } = ev;
  const kind  = ev.kind || "model_deviation";
  const meta  = ANOM_KINDS[kind] || ANOM_KINDS.model_deviation;
  const z     = typeof ev.z_score === "number" ? ev.z_score : null;
  const col   = meta.col;      // each anomaly type has its own unique colour
  const txt   = meta.txt;

  // Reason text: model uses z-derived wording; physical detectors send their own.
  let reason = ev.reason || meta.label;
  if (kind === "model_deviation") {
    if      (z != null && z > 6)   reason = "Extreme deviation — possible spoofing";
    else if (z != null && z > 4.5) reason = "Large trajectory deviation";
    else                           reason = "Off predicted path";
    if (step != null) reason += ` at step ${step + 1}`;
  }

  const entries = document.getElementById("sim-feed-entries");
  let entry = _feedByMmsi.get(mmsi);
  if (!entry) {
    entry = { el: document.createElement("div"), kinds: {}, instances: {} };
    entry.el.className = "feed-entry";
    entry.el.dataset.mmsi = mmsi;
    entry.el.onclick = () => focusVessel(mmsi);
    _feedByMmsi.set(mmsi, entry);
  }
  entry.kinds[kind] = (entry.kinds[kind] || 0) + 1;
  // Retain where/when this alert kind last fired so its chip can jump to it.
  if (ev.lat != null && ev.lon != null) {
    entry.instances[kind] = { lat: ev.lat, lon: ev.lon, sim_time, reason,
                              label: meta.label, col: meta.col };
  }
  const total = Object.values(entry.kinds).reduce((a, b) => a + b, 0);

  // Global tallies: per-kind (filter chip counts) and overall feed total.
  _kindTotals[kind] = (_kindTotals[kind] || 0) + 1;
  const chipN = document.querySelector(`.ac-n[data-count="${kind}"]`);
  if (chipN) chipN.textContent = _kindTotals[kind];
  const feedTotal = document.getElementById("feed-total");
  if (feedTotal) feedTotal.textContent =
    "(" + Object.values(_kindTotals).reduce((a, b) => a + b, 0) + ")";

  // A colour-filled chip per distinct alert kind this ship has triggered.
  // If we know where that kind last fired, the chip is clickable and jumps
  // the map to that exact spot.
  const chips = Object.entries(entry.kinds).map(([k, n]) => {
    const m = ANOM_KINDS[k] || ANOM_KINDS.model_deviation;
    const hasInst = entry.instances && entry.instances[k];
    return `<span class="feed-chip${hasInst ? " clickable" : ""}" style="--c:${m.col}"
              title="${hasInst ? "Show where this fired" : m.label}"
              ${hasInst ? `onclick="event.stopPropagation();jumpToAlertInstance(${mmsi},'${k}')"` : ""}>${m.label}${n > 1 ? " ×" + n : ""}</span>`;
  }).join("");
  const zText = z != null && kind === "model_deviation"
    ? `latest z = <span class="feed-z">${z.toFixed(2)}</span><br>` : "";

  entry.el.style.borderLeftColor = col;
  entry.el.innerHTML =
    `<span class="feed-total-badge" style="--c:${col}" title="${total} alert${total > 1 ? "s" : ""}">×${total}</span>` +
    `<span class="feed-mmsi">${GROUP_NAMES[ship_type] || "Vessel"} ${mmsi}</span>` +
    ` <span class="feed-time">${sim_time}</span>` +
    `<div class="feed-chips">${chips}</div>` +
    `${zText}<span class="feed-reason">${reason}</span>`;

  _applyFeedVisibility(entry);          // honour the active alert-type filter

  // Move (or insert) at the top; restart the fade-in so the update is noticeable.
  entries.insertBefore(entry.el, entries.firstChild);
  entry.el.style.animation = "none";
  void entry.el.offsetWidth;           // reflow to restart the CSS animation
  entry.el.style.animation = "";

  while (entries.children.length > 30) {
    const last = entries.lastChild;
    _feedByMmsi.delete(+last.dataset.mmsi);
    entries.removeChild(last);
  }
}

// Jump the map to the exact spot an alert last fired for a vessel, drop a
// pulse marker and open a popup with the reason + timestamp. Non-destructive:
// playback keeps its place.
// Drop a pulse marker + popup at an alert instance and fly there. Shared by
// the Live Sim feed and the Replay anomaly feed. `key` identifies the instance
// so callers can toggle it off on a repeat click.
function _showInstanceMarker(key, lat, lon, col, label, reason, sub) {
  _clearAlertMarker();
  _alertMarker = L.marker([lat, lon], {
    icon: L.divIcon({
      className: "alert-instance",
      html: `<div class="ai-pulse" style="--c:${col}"></div>`,
      iconSize: [20, 20], iconAnchor: [10, 10],
    }),
    zIndexOffset: 2000,
  }).addTo(map);
  _alertMarker.bindPopup(
    `<div style="min-width:150px">
       <b style="color:${col}">&#9888; ${label}</b><br>${reason}<br>
       <small style="color:#94a3b8">${sub}</small>
     </div>`,
    { closeButton: true, autoClose: false }
  ).openPopup();
  // Closing the popup (✕) also removes the pulse marker so nothing lingers.
  _alertMarker.on("popupclose", _clearAlertMarker);
  _alertMarkerKey = key;
  map.flyTo([lat, lon], Math.max(map.getZoom(), 12), { duration: 0.6 });
}

function _clearAlertMarker() {
  if (_alertMarker) { map.removeLayer(_alertMarker); _alertMarker = null; }
  _alertMarkerKey = null;
}

// Live Sim: jump to where an alert kind last fired for a vessel.
function jumpToAlertInstance(mmsi, kind) {
  const key = `sim:${mmsi}:${kind}`;
  if (_alertMarkerKey === key) { _clearAlertMarker(); return; }   // toggle off
  const entry = _feedByMmsi.get(mmsi);
  const inst  = entry && entry.instances && entry.instances[kind];
  if (!inst) return;
  _showInstanceMarker(key, inst.lat, inst.lon, inst.col, inst.label, inst.reason,
                      `${inst.sim_time}<br>${inst.lat.toFixed(4)}, ${inst.lon.toFixed(4)}`);
}

// Replay: jump to where/when a vessel's anomaly peaked, seek there and mark it.
function jumpToReplayAnomaly(id) {
  const s = ships[id];
  if (!s) return;
  if (_alertMarkerKey === `replay:${id}`) { _clearAlertMarker(); return; }   // toggle off
  // Jump to the earliest active alert (model or rule) with a location.
  const al = _vesselAlerts(s).filter(a => _activeReplayKinds.has(a.kind) && a.lat != null)
                             .sort((a, b) => a.tick - b.tick);
  const a = al[0];
  if (!a) { jumpToShip(id); return; }
  const m = REPLAY_KINDS[a.kind];
  selectReplayShip(id);
  seekTo(a.tick);
  _showInstanceMarker(`replay:${id}`, a.lat, a.lon, m.col,
                      `${s.type_name} — ${m.label}`, a.reason,
                      `step ${a.tick} · ${a.lat.toFixed(4)}, ${a.lon.toFixed(4)}`);
}

function onSimStatus(ev) {
  if (ev.state === "seeking") { _setSeeking(true); }
  else if (seeking && (ev.seeked || ev.state === "running" || ev.state === "paused"
                       || ev.state === "done")) {
    _setSeeking(false);
  }

  applySimStatus(ev);

  if (ev.n_pings       !== undefined) document.getElementById("sim-pings-disp").textContent   = ev.n_pings.toLocaleString();
  if (ev.n_vessels     !== undefined) document.getElementById("sim-vessels-disp").textContent = ev.n_vessels;
  if (ev.n_anomalies   !== undefined) document.getElementById("sim-anom-disp").textContent    = ev.n_anomalies;
  if (ev.n_predictions !== undefined) document.getElementById("sim-pred-disp").textContent    = ev.n_predictions;

  if (!scrubbing && !seeking && typeof ev.progress === "number") {
    document.getElementById("sim-progress").value = Math.round(ev.progress * 1000);
    document.getElementById("sim-progress-pct").textContent = (ev.progress * 100).toFixed(1) + "%";
  }

  document.getElementById("st-total").textContent = ev.n_vessels ?? simVessels.size;
  document.getElementById("st-anom").textContent  = (ev.n_anom_ships ?? 0).toLocaleString();   // distinct ships flagged
  document.getElementById("st-ade").textContent   = (ev.n_anomalies ?? 0).toLocaleString();    // total anomaly events
}


// ══════════════════════════════════════════════════════════════════════════════
//  VESSEL SELECTION  (sim mode)
// ══════════════════════════════════════════════════════════════════════════════

function selectSimVessel(mmsi) {
  if (selectedMmsi === mmsi) { deselectSimVessel(); return; }
  deselectSimVessel();
  selectedMmsi = mmsi;
  const v = simVessels.get(mmsi);
  if (!v) return;

  selHistLine = L.polyline(v.history, { color: v.color, weight: 1.5, opacity: 0.22 }).addTo(map);
  _scheduleIconUpdate(mmsi);
  _flushDirtyIcons();
  _restylePredLines(mmsi);
  _drawSelStepMarkers(mmsi);

  document.getElementById("sel-panel").classList.add("open");
  refreshSelPanel(mmsi);
  map.panTo(v.marker.getLatLng(), { animate: true });
}

function deselectSimVessel() {
  if (!selectedMmsi) return;
  const saved = selectedMmsi;
  selectedMmsi = null;
  const v = simVessels.get(saved);
  if (v) {
    _scheduleIconUpdate(saved);
    _flushDirtyIcons();
    _restylePredLines(saved);
  }
  if (selHistLine) { map.removeLayer(selHistLine); selHistLine = null; }
  _clearSelStepMarkers();
  document.getElementById("sel-panel").classList.remove("open");
}

// One close handler for both modes' selection panels.
function deselectAny() {
  if (simMode) deselectSimVessel();
  else deselectReplayShip();
}

function _newestPred(v) {
  return v && v.predLines.length ? v.predLines[v.predLines.length - 1] : null;
}

function _drawSelStepMarkers(mmsi) {
  _clearSelStepMarkers();
  const v  = simVessels.get(mmsi);
  const np = _newestPred(v);
  if (!np) return;
  np.coords.forEach((pos, i) => {
    const sog = np.sogs && np.sogs[i] !== undefined ? np.sogs[i].toFixed(1) : "—";
    const m = L.circleMarker(pos, {
      radius: 5, fillColor: "#111", fillOpacity: 0.85, color: v.color, weight: 2,
    })
      .bindTooltip(
        `<b>Step ${i + 1}</b><br>${pos[0].toFixed(4)}°N &nbsp; ${pos[1].toFixed(4)}°E<br>SOG ${sog} kn`,
        { sticky: false, direction: "top" })
      .addTo(map);
    selStepMarkers.push(m);
  });
  const origin = v.trail[v.trail.length - 1];
  if (origin) {
    selOriginMarker = L.circleMarker(origin, {
      radius: 7, fillColor: v.color, fillOpacity: 1.0, color: "#fff", weight: 2,
    })
      .bindTooltip("Prediction start", { direction: "top" })
      .addTo(map);
  }
}

function _clearSelStepMarkers() {
  selStepMarkers.forEach(m => map.removeLayer(m));
  selStepMarkers = [];
  if (selOriginMarker) { map.removeLayer(selOriginMarker); selOriginMarker = null; }
}

function refreshSelPanel(mmsi) {
  const v = simVessels.get(mmsi);
  if (!v) return;
  const anom = v.lastAnomaly;
  const anomHtml = anom !== null
    ? `<div class="sel-row"><span>Z-score</span>
         <span class="sel-val" style="color:${anom > 3 ? "#ef4444" : anom > 1.5 ? "#f97316" : "#22c55e"}">
           ${anom.toFixed(2)} σ</span></div>` : "";
  const watchHtml = v.nPings < SEQ_ENC
    ? `<div class="sel-watch">Warming up… ${v.nPings} / ${SEQ_ENC} pings before first prediction</div>` : "";
  const pendingHtml = v.inferPending
    ? `<div class="sel-watch" style="color:#f3ca86">⚡ Inference running…</div>` : "";
  const lastPredHtml = v.lastPredAt
    ? `<div class="sel-row"><span>Last prediction</span><span class="sel-val">${v.lastPredAt}</span></div>` : "";

  const np = _newestPred(v);
  let predHtml;
  if (np) {
    predHtml = `<div class="sel-pred-title">Latest predicted route (${np.coords.length} steps)</div>
      <div class="sel-pred-list">` +
      np.coords.map((pos, i) => {
        const sog = np.sogs && np.sogs[i] !== undefined ? np.sogs[i].toFixed(1) : "—";
        return `<div class="sel-pred-row">
          <span class="sel-step">${i + 1}</span>
          <span>${pos[0].toFixed(3)}°N ${pos[1].toFixed(3)}°E</span>
          <span class="sel-pred-sog">${sog} kn</span></div>`;
      }).join("") + `</div>`;
  } else {
    predHtml = `<div class="sel-watch">No prediction yet${v.nPings >= SEQ_ENC ? " — running…" : ""}</div>`;
  }

  let cmpHtml = "";
  if (v.predLines.length > 1 && np) {
    const rows = [];
    for (let i = v.predLines.length - 2; i >= 0; i--) {
      const older = v.predLines[i];
      const agePreds = (v.predLines.length - 1) - i;
      const k = Math.min(older.coords.length, np.coords.length);
      let sum = 0;
      for (let s = 0; s < k; s++) sum += _haversineKm(older.coords[s], np.coords[s]);
      const drift = k ? sum / k : 0;
      const col = drift > 3 ? "#ef4444" : drift > 1 ? "#f97316" : "#22c55e";
      rows.push(`<div class="sel-cmp-row">
        <span class="sel-cmp-age">${agePreds} pred${agePreds > 1 ? "s" : ""} ago</span>
        <span class="sel-cmp-drift" style="color:${col}">${drift.toFixed(2)} km</span></div>`);
    }
    cmpHtml = `<div class="sel-cmp-title">Prediction drift vs latest</div>${rows.join("")}`;
  }

  document.getElementById("sel-mmsi").textContent = mmsi;
  document.getElementById("sel-type").textContent = GROUP_NAMES[v.ship_type] || "Unknown";
  document.getElementById("sel-body").innerHTML =
    `<div class="sel-row"><span>SOG</span><span class="sel-val">${v.lastSog.toFixed(1)} kn</span></div>
     <div class="sel-row"><span>COG</span><span class="sel-val">${v.lastCog.toFixed(0)}°</span></div>
     <div class="sel-row"><span>Live predictions</span><span class="sel-val">${v.predLines.length}</span></div>
     ${lastPredHtml}${anomHtml}${watchHtml}${pendingHtml}${predHtml}${cmpHtml}`;
}

map.on("click", () => { deselectAny(); });
