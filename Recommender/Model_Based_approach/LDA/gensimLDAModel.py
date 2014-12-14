#!/usr/bin/python
"""Simple script using gensim to extract topics from some training data"""

import json
import gensim.models.ldamulticore as mLDA

if __name__ == "__main__":

    DATA_PATH = "/home/mat/Documents/Git/GroovesharkRecommender/Recommender/Data_processing/f_medium2_corpus.json"
    ALPHA = 0.01
    BETA = 0.5
    NUM_TOPICS = 40

    #Reading the json corpus
    with open(DATA_PATH, 'r') as jsonStream:
        DATA = json.load(jsonStream)

    #Before handing the corpus to gensim we format it to be a list of playlists
    TRAINING_DATA = [DATA["corpus"][idx] for idx in DATA["corpus"]] #transform the dict into a list
    for idx, playlist in enumerate(TRAINING_DATA):    #transform each list into list of tuples
        TRAINING_DATA[idx] = [tuple(x) for x in playlist]

    #The lexicon maps from the ids of the words to their name
    LEXICON = DATA["invLexicon"]

    print "corpus size: ", len(TRAINING_DATA), "playlists  | ", len(LEXICON), "artists\n"

    MODEL = mLDA.LdaMulticore(TRAINING_DATA, num_topics=NUM_TOPICS, id2word=LEXICON \
        , eta=BETA, alpha=ALPHA, workers=3)

    #Print the topics and save the models
    FSTREAM = open("topics.txt", 'w')
    for topic in MODEL.print_topics(NUM_TOPICS, 20):
        FSTREAM.write(topic.encode('utf8'))
        FSTREAM.write("\n\n")

    MODEL.save("./saved_model/lda_model")

    """
    Example of topics extracted with gensim.

    For each topic, what you see is the probability of the 20 top artists:

    0.038*Muse + 0.022*The Black Keys + 0.021*The Strokes + 0.019*Death Cab for Cutie + 
    0.019*Yeah Yeah Yeahs + 0.015*Arctic Monkeys + 0.013*The White Stripes + 0.012*Radiohead + 
    0.012*Vampire Weekend + 0.010*MGMT + 0.009*Two Door Cinema Club + 0.009*Mumford & Sons + 
    0.008*Marilyn Manson + 0.008*Phoenix + 0.008*Passion Pit + 0.007*Modest Mouse + 
    0.007*Cold War Kids + 0.007*Kings of Leon + 0.007*Florence and The Machine + 0.007*Regina Spektor

    0.052*Enya + 0.043*The Doors + 0.041*The Cramps + 0.032*The Who + 0.023*The Kinks + 
    0.018*The Rolling Stones + 0.016*Grateful Dead + 0.016*Neil Young + 0.015*Ramones + 
    0.015*Bob Dylan + 0.010*Frank Sinatra + 0.009*Tangerine Dream + 0.009*Clannad + 
    0.009*Los Cafres + 0.009*The Misfits + 0.008*Led Zeppelin + 0.008*Christmas + 
    0.008*Jimi Hendrix + 0.007*Cultura Profética + 0.007*Yes

    0.196*Jack Johnson + 0.076*The Ventures + 0.054*John Mayer + 0.051*Mac Miller + 0.024*Adele + 
    0.016*Carpenters + 0.016*Jason Mraz + 0.012*Va - www.musicasparabaixar.org + 
    0.010*Alkaline Trio + 0.010*Silvio Rodríguez + 0.010*Tyga + 0.008*Rod Stewart + 
    0.007*Bing Crosby + 0.007*2 Chainz + 0.007*De La Soul + 0.006*Frank Sinatra + 0.006*Chinoy + 
    0.006*Gavin DeGraw + 0.006*Future + 0.005*Dean Martin

    0.099*The Rolling Stones + 0.043*Frank Sinatra + 0.032*Ella Fitzgerald + 0.025*John Coltrane + 
    0.022*Charlie Parker + 0.020*Louis Armstrong + 0.018*Miles Davis + 0.015*Billie Holiday + 
    0.014*Chick Corea + 0.014*Nina Simone + 0.011*Elvis Presley + 0.011*Frank Zappa + 
    0.011*Bob James + 0.009*Incognito + 0.008*Diana Krall + 0.008*Ed Sheeran  + 0.008*Ray Charles + 
    0.007*Dizzy Gillespie + 0.007*Cesária Évora + 0.006*Pat Metheny

    0.019*Casting Crowns + 0.013*Various + 0.012*Chris Tomlin + 0.012*Explosions in the Sky + 
    0.012*Bajofondo + 0.012*Third Day + 0.011*NEEDTOBREATHE + 0.011*tobyMac + 0.010*Skillet + 
    0.010*Hillsong United + 0.010*Mogwai + 0.010*Chris Rice + 0.009*MercyMe + 0.009*Hillsong + 
    0.008*Lewis Black + 0.008*BarlowGirl + 0.008*Jesus Culture + 0.008*Fred Hammond + 
    0.008*Matt Redman + 0.007*Phil Wickham

    0.177*Various Artists + 0.076*Dropkick Murphys + 0.046*The Pogues + 0.044*Dave Matthews Band + 
    0.041*Flogging Molly + 0.024*Brand New + 0.023*Phish + 0.019*The Dubliners + 
    0.015*A. R. Rahman + 0.014*The Irish Rovers + 0.013*Jimmy Buffett + 0.012*O.A.R. + 
    0.012*Dispatch + 0.008*Guster + 0.008*Angels & Airwaves + 0.007*Blackmore's Night + 0.007*Boa + 
    0.007*The Tossers + 0.006*The Chieftains + 0.006*Dave Matthews

    0.146*Glee Cast + 0.060*Naruto + 0.042*These Animal Men + 0.028*梶浦由記 + 0.027*Glee + 
    0.013*Gravitation + 0.011*Paolo Nutini + 0.009*Ion Storm + 0.009*See-Saw + 
    0.008*Straight No Chaser + 0.008*Tori Amos + 0.008*Bruce Faulconer + 0.008*???Y ?R?L + 
    0.007*Mormon Tabernacle Choir + 0.007*Fruits Basket + 0.006*Gundam Wing + 0.006*Shoji Meguro + 
    0.006*Various Artists + 0.005*鷺巣詩郎 + 0.005*Elfen Lied

    0.061*Hans Zimmer + 0.052*John Williams + 0.039*Ion Storm + 0.035*James Horner + 
    0.034*Howard Shore + 0.032*Daddy Yankee + 0.020*Wisin y Yandel + 0.016*CéU + 0.015*Don Omar + 
    0.014*Jeremy Soule + 0.011*James Newton Howard + 0.011*Relient K + 0.011*Steve Jablonsky + 
    0.010*Danny Elfman + 0.009*Rakim y Ken-Y + 0.009*Clint Mansell + 
    0.008*Hans Zimmer & James Newton Howard + 0.007*Alexandre Desplat + 
    0.007*The City of Prague Philharmonc + 0.007*Thomas Newman

    0.026*Pearl Jam + 0.022*Linkin Park + 0.020*Green Day + 0.020*Foo Fighters + 0.014*Pantera + 
    0.014*Red Hot Chili Peppers + 0.012*Iron Maiden + 0.010*Metallica + 0.010*Soundgarden + 
    0.009*Children of Bodom + 0.009*Counting Crows + 0.009*Van Halen + 0.008*Megadeth + 
    0.008*Helloween + 0.008*Slayer + 0.008*Infected Mushroom + 0.008*Rush + 
    0.007*Stone Temple Pilots + 0.007*Yiruma + 0.007*Nightwish

    0.033*Guns N' Roses + 0.027*AC/DC + 0.025*Jimi Hendrix + 0.025*Creedence Clearwater Revival + 
    0.017*Michael Jackson + 0.017*Eric Clapton + 0.017*Led Zeppelin + 0.016*The Rolling Stones + 
    0.013*Aerosmith + 0.013*Steve Miller Band + 0.013*Lynyrd Skynyrd + 0.012*Def Leppard + 
    0.011*Bon Jovi + 0.010*Fleetwood Mac + 0.009*Styx + 0.009*Red Hot Chili Peppers + 0.009*Queen + 
    0.009*The Who + 0.008*Van Halen + 0.007*The Police

    0.137*Bob Marley + 0.051*Tiësto + 0.037*Damian Marley + 0.030*Atmosphere + 
    0.025*Alanis Morissette + 0.025*Ray LaMontagne + 0.019*Armin van Buuren + 0.013*Nightwish + 
    0.012*Snoop Dogg + 0.012*Beenie Man + 0.012*Talking Heads + 0.011*Sage Francis + 
    0.010*Vybz Kartel + 0.009*Sheryl Crow + 0.009*Tina Dickow + 0.009*David Gray + 
    0.008*Brother Ali + 0.007*Coldplay + 0.007*Alexandre Desplat + 0.007*Front Line Assembly

    0.061*Avenged Sevenfold + 0.042*Sublime + 0.025*Limp Bizkit + 0.024*Papa Roach + 
    0.017*Slipknot + 0.017*Nine Inch Nails + 0.015*Three Days Grace + 0.014*Insane Clown Posse + 
    0.014*Breaking Benjamin + 0.013*Tool + 0.011*Linkin Park + 0.011*Rob Zombie + 
    0.011*Hollywood Undead + 0.010*Shinedown + 0.010*Skillet + 0.010*Escape the Fate + 
    0.009*Godsmack + 0.009*Staind + 0.008*Queen + 0.007*3 Doors Down

    0.022*Tech N9ne + 0.020*Young Jeezy + 0.019*Nelly + 0.019*Ludacris + 0.018*Lil Wayne + 
    0.018*T.I. + 0.017*Akon + 0.017*Rick Ross + 0.016*Stevie Wonder + 0.014*Ja Rule + 
    0.013*T-Pain + 0.012*50 Cent + 0.011*Bone Thugs-n-Harmony + 0.010*Plies + 0.009*Gucci Mane + 
    0.009*Missy Elliott + 0.009*Wu-Tang Clan + 0.008*Lil Boosie + 0.008*Snoop Dogg + 
    0.008*The Notorious B.I.G.

    0.025*CunninLynguists + 0.022*Thievery Corporation + 0.021*Massive Attack + 0.016*Zero 7 + 
    0.015*Teoman + 0.014*Cafe Del Mar + 0.014*Daft Punk + 0.013*Cut Copy + 0.012*Şebnem Ferah + 
    0.010*Morcheeba + 0.009*Tosca + 0.009*Portishead + 0.008*Hot Chip + 0.007*Café Del Mar + 
    0.007*Amon Tobin + 0.007*Lamb + 0.007*Duman + 0.007*St. Germain + 0.006*Groove Armada + 
    0.006*El-P

    0.026*Don Omar + 0.021*Maná + 0.020*Luis Miguel + 0.017*Calle 13 + 0.016*Wisin y Yandel + 
    0.016*Never Shout Never + 0.015*Silvio Rodríguez + 0.013*Dire Straits + 0.013*Juanes + 
    0.012*Muddy Waters + 0.012*Joaquín Sabina + 0.012*B.B. King + 0.011*Silvestre Dangond + 
    0.011*John Lee Hooker + 0.009*Morodo + 0.009*Alejandro Sanz + 0.009*Gondwana + 
    0.009*Andrés Cepeda + 0.009*Aventura + 0.008*Howlin' Wolf

    0.057*Elton John + 0.021*Cat Stevens + 0.017*Steely Dan + 0.014*植松伸夫 + 0.014*Weezer + 
    0.013*Paul Simon + 0.013*Simon & Garfunkel + 0.012*James Taylor + 0.012*Eagles + 0.011*Bread + 
    0.011*The Moody Blues + 0.010*The Mamas & the Papas + 0.009*Chicago + 0.008*Jim Croce + 
    0.008*Peter, Paul & Mary + 0.007*Mägo de Oz + 0.007*Three Dog Night + 0.006*Billy Joel + 
    0.006*The Shins + 0.006*The Doobie Brothers

    0.043*Bob Marley & The Wailers + 0.033*U2 + 0.030*The Cure + 0.021*The Smiths + 
    0.017*Iron & Wine + 0.014*Tom Waits + 0.013*Bob Marley + 0.012*Fleet Foxes + 
    0.009*Stereophonics + 0.008*Joy Division + 0.008*Stars + 0.007*They Might Be Giants + 
    0.007*The Velvet Underground + 0.007*Wilco + 0.007*Tracy Chapman + 0.006*Björk + 
    0.006*Ryan Adams + 0.006*Keane + 0.006*Beirut + 0.005*The Flaming Lips

    0.067*Disturbed + 0.060*Loreena McKennitt + 0.036*Immortal Technique + 0.036*Feist + 
    0.027*Andrew Bird + 0.026*Tegan and Sara + 0.023*Cradle of Filth + 0.023*Bright Eyes + 
    0.023*Dead Can Dance + 0.019*Metric + 0.018*Mac Dre + 0.015*Easy Star All-Stars + 
    0.013*Sufjan Stevens + 0.013*The String Quartet + 0.009*Beats Antique + 
    0.008*Camarón de la Isla + 0.008*Joe Strummer + 0.007*Soulfly + 0.007*Lisa Gerrard + 
    0.006*Battlefield Band

    0.131*Metallica + 0.087*50 Cent + 0.030*Bring Me the Horizon + 0.021*Toots & The Maytals + 
    0.021*Los Amigos Invisibles + 0.018*Moulin Rouge + 0.018*Parkway Drive + 0.013*John Fogerty + 
    0.012*Ska Cubano + 0.011*Rodrigo + 0.010*The Muppets + 0.009*Parokya ni Edgar + 
    0.009*Umphrey's McGee + 0.009*HIM + 0.009*The Countdown Kids + 0.008*As I Lay Dying + 
    0.008*Robert Pattinson + 0.008*Ska-P + 0.008*The Skatalites + 0.007*H.I.M.

    0.034*Maroon 5 + 0.032*Rihanna + 0.024*Nicki Minaj + 0.024*Drake + 0.020*Kid Cudi + 
    0.017*Eminem + 0.015*Chris Brown + 0.013*Foster The People + 0.013*Flo Rida + 
    0.013*Bruno Mars + 0.011*Florence and the Machine + 0.011*Red Hot Chili Peppers + 
    0.011*Lady Gaga + 0.010*OneRepublic + 0.010*LMFAO + 0.010*The Black Eyed Peas + 
    0.010*The Black Keys + 0.009*Usher + 0.009*Beyoncé + 0.009*Lupe Fiasco

    0.094*Rihanna + 0.048*Jason Mraz + 0.026*Madonna + 0.024*Christina Aguilera + 0.024*Lady Gaga + 
    0.022*The Black Eyed Peas + 0.018*Katy Perry + 0.017*Ricardo Arjona + 0.017*Beyoncé + 
    0.014*The Pussycat Dolls + 0.014*Pitbull + 0.014*Enrique Iglesias + 0.014*Britney Spears + 
    0.012*Jennifer Lopez + 0.011*Disney + 0.010*Shakira + 0.008*Mika + 0.007*Alejandro Sanz + 
    0.007*Fergie + 0.007*Gwen Stefani

    0.025*The Notorious B.I.G. + 0.024*Vitamin String Quartet + 0.012*A. R. Rahman + 
    0.011*South Park Mexican + 0.010*Cross Canadian Ragweed + 0.010*Best Coast + 0.010*India.Arie + 
    0.009*Jim Sturgess + 0.009*Pretty Lights + 0.009*Janelle Monae + 0.008*Randy Rogers Band + 
    0.008*Across The Universe + 0.007*Sleigh Bells + 0.007*toro y moi + 0.007*Ween + 
    0.007*सोनू निगम + 0.007*Tupac Shakur + 0.007*Tupac + 0.007*Bo Burnham + 0.006*The Roots

    0.057*blink-182 + 0.019*A Day to Remember + 0.018*NOFX + 0.014*The Offspring + 0.014*Sum 41 + 
    0.014*3 Doors Down + 0.013*Green Day + 0.013*Mayday Parade + 0.013*My Chemical Romance + 
    0.012*Garbage + 0.012*Rancid + 0.011*Paramore + 0.011*The All-American Rejects + 
    0.011*Boyce Avenue + 0.010*New Found Glory + 0.010*Simple Plan + 0.010*Fall Out Boy + 
    0.009*Bad Religion + 0.009*Taking Back Sunday + 0.009*Dashboard Confessional

    0.419*Lil Wayne + 0.033*Akon + 0.023*Kid Cudi + 0.021*T.I. + 0.018*Nicki Minaj + 
    0.015*Gucci Mane + 0.013*Chris Brown + 0.013*Young Jeezy + 0.013*Eminem + 0.012*Britney Spears + 
    0.011*DJ Khaled + 0.011*Lil Boosie + 0.010*The Game + 0.009*Kevin Rudolf + 0.009*50 Cent + 
    0.008*Damien Rice + 0.007*Birdman + 0.007*Young Money + 0.006*Drake + 0.006*T-Pain

    0.251*Eminem + 0.107*2Pac + 0.043*Dr. Dre + 0.027*50 Cent + 0.024*Fall Out Boy + 0.018*Moby + 
    0.017*Tupac + 0.015*2.PAC + 0.014*Snoop Dogg + 0.013*The Notorious B.I.G. + 
    0.012*Secret Garden + 0.011*The Game + 0.010*Panic! At the Disco + 0.008*Nate Dogg + 
    0.008*D12 + 0.006*Buddha Bar (CD Series) + 0.006*brokeNCYDE + 0.005*Mike Oldfield + 
    0.005*Tech N9ne + 0.005*Warren G

    0.152*Drake + 0.047*Matisyahu + 0.041*Wiz Khalifa + 0.031*Aventura + 0.021*J. Cole + 
    0.019*Black Uhuru + 0.018*Wale + 0.018*Gucci Mane + 0.017*Alpha Blondy + 0.017*Jimmy Cliff + 
    0.016*Tiken Jah Fakoly + 0.014*King Tubby + 0.013*Jorge Drexler + 0.010*Fabolous + 
    0.010*Waka Flocka Flame + 0.010*Big Sean + 0.010*Mercedes Sosa + 0.008*Tyga + 
    0.008*Chris Brown + 0.007*Shwayze

    0.079*Kanye West + 0.064*Jay-Z + 0.049*Michael Bublé + 0.042*Original Broadway Cast + 
    0.027*Jonas Brothers + 0.020*Nas + 0.016*Jonathan Larson + 0.016*Wicked + 0.016*Les Miserables + 
    0.014*Andrew Lloyd Webber + 0.012*Timbaland + 0.012*Lupe Fiasco + 0.011*DMX + 
    0.010*Various Artists + 0.009*Snoop Dogg + 0.008*Jadakiss + 0.008*The Notorious B.I.G. + 
    0.007*Miley Cyrus + 0.007*A Tribe Called Quest + 0.007*Kid Cudi

    0.043*Tim McGraw + 0.032*George Strait + 0.029*Carrie Underwood + 0.027*Rascal Flatts + 
    0.026*Toby Keith + 0.025*Jason Aldean + 0.023*Taylor Swift + 0.022*Zac Brown Band + 
    0.018*Blake Shelton + 0.018*Sugarland + 0.017*Kenny Chesney + 0.017*Randy Travis + 
    0.016*Josh Turner + 0.016*Alan Jackson + 0.016*Fleetwood Mac + 0.015*Brad Paisley + 
    0.012*Billy Currington + 0.011*Gary Allan + 0.011*Regina Spektor + 0.011*The Band Perry

    0.027*Techno + 0.019*Kirk Franklin + 0.016*Groove Coverage + 0.015*Björk + 0.014*Lounge + 
    0.013*Gotan Project + 0.012*V A + 0.012*Buddha-Bar + 0.011*Ursula 1000 + 0.010*Tiësto + 
    0.010*Caetano Veloso + 0.009*Gigi D'Agostino + 0.009*Paul Oakenfold + 0.008*Scooter + 
    0.008*Chico Buarque + 0.008*Cascada + 0.007*Stephen Lynch + 0.006*Paul van Dyk + 
    0.006*Benny Benassi + 0.006*Gal Costa

    0.057*Black Sabbath + 0.051*Girl Talk + 0.048*Backstreet Boys + 0.025*RJD2 + 
    0.024*Parov Stelar + 0.017*DJ Shadow + 0.017*Ennio Morricone + 0.015*mc chris + 
    0.014*Wax Tailor + 0.013*Chico Trujillo + 0.012*Ladytron + 0.010*Thin Lizzy + 
    0.010*John Pizzarelli + 0.009*Rita Lee + 0.009*Tyler, The Creator + 
    0.008*Brazilian Tropical Orchestra + 0.008*RJD2 + 0.007*Pogo + 0.007*Squirrel Nut Zippers + 
    0.007*Caravan Palace

    0.122*Frédéric Chopin + 0.117*Johann Sebastian Bach + 0.076*Wolfgang Amadeus Mozart + 
    0.053*Ludwig van Beethoven + 0.051*Antonio Vivaldi + 0.032*Chopin + 0.030*Beethoven + 
    0.026*Sting + 0.024*Led Zeppelin + 0.021*Mozart + 0.020*Glenn Gould + 0.016*Vladimir Horowitz + 
    0.014*Horowitz, Vladimir + 0.014*Georg Friedrich Händel + 0.014*Tchaikovsky + 0.013*Bach + 
    0.011*Alfred Brendel + 0.010*Johannes Brahms + 0.009*B A C H + 0.009*Dream Theater

    0.075*Justin Bieber + 0.053*Eminem + 0.040*Florence and The Machine + 
    0.037*The Black Eyed Peas + 0.031*Mumford & Sons + 0.030*Kid Cudi + 0.029*Usher + 
    0.029*Chris Brown + 0.026*Rihanna + 0.022*B.o.B + 0.022*Lil Wayne + 0.018*The Lonely Island + 
    0.017*Flo Rida + 0.016*Kanye West + 0.015*Pink + 0.015*Owl City + 0.014*Ingrid Michaelson + 
    0.013*Far East Movement + 0.013*Skillet + 0.012*Glee Cast

    0.042*Two Steps From Hell + 0.040*Bach + 0.033*ABBA + 0.030*Amy Winehouse + 
    0.028*Claude Debussy + 0.023*Whitney Houston + 0.018*Buena Vista Social Club + 
    0.014*Stravinsky + 0.014*Yo-Yo Ma + 0.009*Crazy Frog + 0.009*B A C H + 
    0.008*The Brand New Heavies + 0.008*Jacques Brel + 0.008*Audiomachine + 0.008*Bach + 
    0.008*Immediate Music + 0.008*Johann Sebastian Bach + 0.007*Abandon All Ships + 
    0.007*A Skylit Drive + 0.007*Tchaikovsky

    0.021*Benga + 0.020*The Police + 0.019*Skream + 0.018*Queen + 0.017*Bar 9 + 0.014*La Roux + 
    0.013*U 2 + 0.012*Tears for Fears + 0.012*Rusko + 0.011*DJ /rupture + 0.010*Scissor Sisters + 
    0.009*80s + 0.009*The Brian Setzer Orchestra + 0.009*Erasure + 0.008*Radiohead + 0.008*Caspa + 
    0.008*James + 0.007*Madonna + 0.007*Caspa & Rusko + 0.007*Datsik

    0.089*The Killers + 0.045*Evanescence + 0.032*Pendulum + 0.023*"Weird Al" Yankovic + 
    0.023*Joaquín Sabina + 0.021*五月天 + 0.019*Jamiroquai + 0.019*Putumayo + 0.017*Snow Patrol + 
    0.017*王力宏 + 0.015*Jay Chou 周杰伦 + 0.014*Rage Against the Machine + 0.013*Ray Charles + 
    0.013*林俊傑 + 0.013*KISS + 0.012*Faith No More + 0.010*張學友 + 0.010*Pendulum + 0.009*陳奕迅 + 
    0.009*陶喆

    0.061*Nirvana + 0.044*소녀시대 + 0.033*Josh Groban + 0.032*Kenny G + 0.024*R.E.M. + 
    0.023*Colbie Caillat + 0.019*The Avett Brothers + 0.015*Nujabes + 0.014*John Legend + 
    0.014*Extremoduro + 0.013*2NE1 + 0.012*Ludovico Einaudi + 0.011*Super Junior + 0.011*Big Bang + 
    0.011*Kara + 0.011*Andrea Bocelli + 0.011*Train + 0.010*BoA + 0.010*Flight of the Conchords + 
    0.008*Epik High

    0.079*Héctor Lavoe + 0.027*Marc Anthony + 0.018*Spanish Harlem Orchestra + 
    0.015*Gilberto Santa Rosa + 0.014*El Gran Combo de Puerto Rico + 0.013*Steve Earle + 
    0.012*Nina Simone + 0.012*Celia Cruz + 0.012*Eddie Santiago + 0.012*Medeski Martin and Wood + 
    0.010*Willie Colón + 0.010*Joe Arroyo + 0.010*Deuter + 0.009*N-Dubz + 0.009*Headhunterz + 
    0.009*Afro-Cuban Jazz Project + 0.009*Frankie Ruiz + 0.009*Grupo Niche + 0.008*Maelo Ruiz + 
    0.008*Fania All-Stars

    0.058*Skrillex + 0.040*deadmau5 + 0.030*Bassnectar + 0.028*Mt Eden + 0.022*The Prodigy + 
    0.018*Calvin Harris + 0.018*Johnny Cash + 0.015*MSTRKRFT + 0.014*Nero + 0.010*La Roux + 
    0.010*Flux Pavilion + 0.009*David Guetta + 0.009*Chromeo + 0.009*Justice + 0.009*Rusko + 
    0.009*Imogen Heap + 0.009*Example + 0.009*Avicii + 0.008*The Glitch Mob + 0.008*Boys Noize

    0.025*Marvin Gaye + 0.021*Jackson 5 + 0.018*Stevie Wonder + 0.015*Bee Gees + 
    0.014*The Temptations + 0.013*Damian Marley + 0.013*Ray Charles + 0.011*Barry White + 
    0.011*Aretha Franklin + 0.009*Otis Redding + 0.009*Keiko Matsui + 0.009*John Zorn + 
    0.008*Al Green + 0.008*Mavado + 0.007*Boney James + 0.007*James Brown + 0.007*Tom Jones + 
    0.007*The Isley Brothers, The Isley Brothers + 0.007*Enigma + 0.007*Chuck Berry

    0.026*Trey Songz + 0.018*Mary J. Blige + 0.018*Mariah Carey + 0.016*Andrés Calamaro + 
    0.013*Usher + 0.013*La Renga + 0.013*Ne-Yo + 0.013*Chris Brown + 0.013*R. Kelly + 
    0.012*Paramore + 0.011*Bersuit Vergarabat + 0.011*Manu Chao + 0.011*Aaliyah + 
    0.010*Destiny child + 0.010*Soda Stereo + 0.010*Los Piojos + 0.010*Alicia Keys + 
    0.010*Patricio Rey y sus Redonditos de Ricota + 0.009*Anthony Hamilton + 0.009*Miranda
    """
    