import numpy as np


def tag_around(tag: str, input: str or float):
    start = '<' + tag + '>'
    end = '</' + tag + '>\n'
    return start + str(input) + end

if __name__ == '__main__':
    
    # Every script must begin and end woth the same tags

    number_of_flashes = 3
    # interflash interval:
    TM_interval = np.ones(number_of_flashes)
    TM_start = 0
    TM_step = 20
    # Duration of each flash
    #e.g. all contant 100 ms
    durations = .100 * np.ones((number_of_flashes)) # seconds
    # another example could be with increasing interpulses intervals
    # and flashes durations
    # TM_interval = np.linspace(0, 40, number_of_flashes)
    # durations = np.linspace(100, 500, number_of_flashes)
    head = '<Cluster>\n'
    head += '<Name>DMD Script Config</Name>\n'
    head += '<NumElts>1</NumElts>\n'
    head += '<Array>\n'
    head += '<Name>[DMD script steps]</Name>\n'
    head += '<Dimsize>' + str(number_of_flashes) + '</Dimsize>\n'

    # Images must follow this template:
    # first flahs would have roi_0.png
    # second flash would have roi_1.png
    # etc.
    # add possibility for same ROI?

    for i in range(number_of_flashes):
        
        flash = '<Cluster>\n'
        flash += tag_around('Name', 'DMD settings for 1 TM')
        flash += '<NumElts>10</NumElts>\n'
        flash += '<String>\n'
        flash += tag_around('Name', 'PNG name')
        # here is where the images are updated
        flash += tag_around('Val', 'roi_' + str(i) + '.png')
        flash += '</String>\n'
        flash += '<I32>\n'
        flash += tag_around('Name', 'TM index')
        #again, 10 is hardcoded and has probabluy a meaning
        flash += tag_around('Val', int(TM_start + i * TM_step))
        flash += '</I32>\n'
        flash += '<SGL>\n'
        flash += tag_around('Name', 'Pre wait (s)') 
        flash += tag_around('Val', 0.00000) 
        flash += '</SGL>\n'
        flash += '<SGL>\n'
        flash += tag_around('Name', 'Pulse (s)') 
        flash += tag_around('Val', durations[i]) 
        flash += '</SGL>\n'
        flash += '<I32>\n'
        flash += tag_around('Name', '# triggers')
        flash += tag_around('Val', 1) # hardcoded for now
        flash += '</I32>\n'
        flash += '<SGL>\n'
        flash += tag_around('Name', 'Period (s)') 
        flash += tag_around('Val', durations[i]) # not really sure why
        flash += '</SGL>\n'
        flash += '<Cluster>\n'
        flash += tag_around('Name', 'Final Offset (pix)') 
        flash += tag_around('NumElts', 2) # again, it probably means something
        flash += '<I32>\n'
        flash += tag_around('Name', 'X')
        flash += tag_around('Val', int(0)) # (>.<) no discernable meaning
        flash += '</I32>\n'
        flash += '<I32>\n'
        flash += tag_around('Name', 'Y')
        flash += tag_around('Val', int(0)) # (>.<) no discernable meaning
        flash += '</I32>\n'
        flash += '</Cluster>\n'
        flash += '<Boolean>\n'
        flash += tag_around('Name', 'Motion?')
        flash += tag_around('Val', int(0))
        flash += '</Boolean>\n'
        flash += '<EW>\n'
        flash += tag_around('Name', 'Offset space')
        flash += tag_around('Choice', 'DMD')
        flash += tag_around('Choice', 'Camera 1')
        flash += tag_around('Choice', 'Camera 2')
        flash += tag_around('Val', int(0))
        flash += '</EW>\n'
        flash += '<SGL>\n'
        flash += tag_around('Name', 'Power Multiplier')
        flash += tag_around('Val', 1.00000)
        flash += '</SGL>\n'
        flash += '/Cluster>\n'

    tail = '</Array>\n'
    tail += '</Cluster>\n'

    with open('') as f:
        f.write()