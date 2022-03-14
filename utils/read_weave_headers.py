from astropy.io import fits

# READ KEYWORDS
red_hdul = fits.open('/home/pablo/WEAVE-Apertiff/cubes/20170930/stackcube_1004097.fit')
blue_hdul = fits.open('/home/pablo/WEAVE-Apertiff/cubes/20170930/stackcube_1004097.fit')

for i in range(len(red_hdul)):
    red_hdr = red_hdul[i].header
    blue_hdr = blue_hdul[i].header

    with open('input/weave_specifics/cube_header/red_header_{}.yml'.format(i), 'w') as f:
        for key in list(red_hdr.keys()):
            value = [red_hdr[key]]
            print(key, str(value))
            f.write(key + ': ' + str(value) + '\n')
            if key == 'DATASUM':
                break
    with open('input/weave_specifics/cube_header/blue_header_{}.yml'.format(i), 'w') as f:
        for key in list(blue_hdr.keys()):
            value = [blue_hdr[key]]
            print(key, str(value))
            f.write(key + ': ' + str(value) + '\n')
            if key == 'DATASUM':
                break