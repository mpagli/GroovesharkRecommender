### Latent Dirichlet Allocation (LDA)

LDA is a method mainly used for topic modeling. It is a generative model. The assumption is that documents are created from the following procedure:

* a distribution over the topic theta is sampled from a dirichlet distribution with hyperparameter alpha.
* to generate each word of the document:
	* a topic z is sampled from theta
	* a word is sampled from P(word|theta) = phi_z. phi_z represent a topic. Topics are drawn from a dirichlet distribution of hyperparameter beta.

The problem is that we initially only have the documents, not the phis and thetas of our model. Two main methods exist to perform the inference, variational inference and gibbs sampling. LDA is not the only generative probabilistic topic modeling method. The great advantage of LDA among others yields in the dirichlet priors allowing us to easily infer a topic mixtures for new documents.

In our setup, each playlist is a document, a word is an artist index. We used two libraries implementing LDA: [gensim](http://radimrehurek.com/gensim/models/ldamulticore.html#module-gensim.models.ldamulticore) (python) and [Mallet](http://mallet.cs.umass.edu/) (Java). Gensim uses variational inference, Mallet uses collapsed gibbs sampling. These two methods aim to estimate the same thing, however mallet by default uses methods to optimize the values of the hypermparameters alpha and beta, which greatly improve the quality of the results.

#### Gensim   

After many test with Gensim, here is my favorite configuration: 
* 40 topics. I tried several values but more topics doesn't seem to yield better results.
* alpha = 0.0001: most of the probability weight of theta are given to few topics i.e. one document is not a mixture of many topics
* beta = 0.5: each topic consists in many artists, we don't have few words that keep all the probability mass for one topic. 
* We are using the medium dataset, it contains playlists of 7 artists at least. In total we are crunching 140629 playlists concerning 9303 artists only. I made some tests with a larger version of the dataset but the results where not satisfying (larger means the documents are shorter). 

With these parameters the assumption is that a playlist contains very few topics, and the probability mass of a topic is distributed among many artists. 

Here are the top words for each topic:

		0.018*Breaking Benjamin 
		0.017*Tool 
		0.017*3 Doors Down 
		0.016*Shinedown 
		0.016*Escape the Fate 
		0.015*Papa Roach 
		0.014*Staind 
		0.013*Avenged Sevenfold 
		0.012*Deftones 
		0.011*Tom Waits 
		0.010*Foo Fighters 
		0.010*Three Days Grace 
		0.010*The Offspring 
		0.009*System of a Down 
		0.009*Bullet for My Valentine 
		0.008*Seether 
		0.008*Coheed and Cambria 
		0.008*Metallica 
		0.008*Atreyu 
		0.008*Nickelback

		0.253*Drake 
		0.042*Wiz Khalifa 
		0.028*J. Cole 
		0.027*Feist 
		0.022*Keiko Matsui 
		0.020*Chris Brown 
		0.020*Wale 
		0.015*Tyga 
		0.013*Big Sean 
		0.013*Portishead 
		0.011*Bright Eyes 
		0.010*Calvin Harris 
		0.008*Kendrick Lamar 
		0.008*Nicolas Jaar 
		0.008*Nicki Minaj 
		0.008*Martin O'Donnell & Michael Salvatori 
		0.008*2 Chainz 
		0.007*CocoRosie 
		0.007*Angelo Badalamenti 
		0.007*Meek Mill

		0.063*Metallica 
		0.051*AC/DC 
		0.030*Pantera 
		0.027*Black Sabbath 
		0.027*Disturbed 
		0.022*Iron Maiden 
		0.019*Slayer 
		0.018*Megadeth 
		0.016*KISS 
		0.015*Nightwish 
		0.015*Van Halen 
		0.014*Helloween 
		0.013*Manowar 
		0.012*Scorpions 
		0.011*Explosions in the Sky 
		0.011*Sonata Arctica 
		0.011*Sepultura 
		0.010*Guns N' Roses 
		0.010*Tangerine Dream 
		0.010*Motörhead

		0.030*Benga 
		0.027*Skream 
		0.025*Ella Fitzgerald 
		0.023*Rusko 
		0.023*La Roux 
		0.023*Bar 9 
		0.019*Diana Krall 
		0.019*Charlie Parker 
		0.017*Louis Armstrong 
		0.015*DJ /rupture 
		0.013*Tiken Jah Fakoly 
		0.013*Etta James 
		0.013*Datsik 
		0.012*Billie Holiday 
		0.011*Nero 
		0.011*Caspa 
		0.010*Beach House 
		0.010*Eva Cassidy 
		0.009*Caspa & Rusko 
		0.008*Rockapella

		0.061*Justin Bieber 
		0.053*Eminem 
		0.037*The Black Eyed Peas 
		0.036*Kid Cudi 
		0.035*Chris Brown 
		0.035*Rihanna 
		0.026*Florence and The Machine 
		0.025*Nicki Minaj 
		0.025*Kanye West 
		0.024*Lil Wayne 
		0.023*Usher 
		0.021*Mumford & Sons 
		0.020*B.o.B 
		0.016*The Lonely Island 
		0.015*Lupe Fiasco 
		0.015*Flo Rida 
		0.015*Pink 
		0.012*Jennifer Lopez 
		0.012*Enrique Iglesias 
		0.012*Far East Movement

		0.047*Michael Jackson 
		0.038*R.E.M. 
		0.036*Teoman 
		0.034*Los Piojos 
		0.029*Şebnem Ferah 
		0.022*La Renga 
		0.014*Tom Petty and The Heartbreakers 
		0.010*Rita Lee 
		0.009*David Gray 
		0.009*Sheryl Crow 
		0.009*John Pizzarelli 
		0.008*Duman 
		0.008*Pedro Aznar 
		0.008*Sarah McLachlan 
		0.008*Tom Petty 
		0.007*Intoxicados 
		0.007*Brazilian Tropical Orchestra 
		0.007*Richie Havens 
		0.006*Sarah Mc Lachlan 
		0.006*Bear McCreary

		0.190*Eminem 
		0.053*Tech N9ne 
		0.043*Limp Bizkit 
		0.035*Gucci Mane 
		0.018*Lil Wayne 
		0.017*Timbaland 
		0.015*Papa Roach 
		0.014*50 Cent 
		0.014*Mariah Carey 
		0.013*Insane Clown Posse 
		0.013*Three Days Grace 
		0.012*R. Kelly 
		0.011*Ashanti 
		0.011*Godsmack 
		0.010*T.I. 
		0.009*Twiztid 
		0.009*South Park Mexican 
		0.009*Waka Flocka Flame 
		0.007*Justin Timberlake 
		0.007*Gorilla Zoe

		0.064*Dropkick Murphys 
		0.036*The Pogues 
		0.033*Flogging Molly 
		0.028*Techno 
		0.017*James Blunt 
		0.015*植松伸夫 
		0.011*The Irish Rovers 
		0.011*The Dubliners 
		0.010*Gigi D'Agostino 
		0.009*Cascada 
		0.009*Apparat 
		0.008*Less Than Jake 
		0.008*Clannad 
		0.007*Against Me! 
		0.007*Reverend Horton Heat 
		0.007*Trentemøller 
		0.007*Various Artists 
		0.007*Basshunter 
		0.007*Paul Kalkbrenner 
		0.006*Reel Big Fish

		0.021*Regina Spektor 
		0.019*Yeah Yeah Yeahs 
		0.014*Vampire Weekend 
		0.011*Iron & Wine 
		0.011*Florence and The Machine 
		0.010*Two Door Cinema Club 
		0.009*Passion Pit 
		0.009*Metric 
		0.008*deadmau5 
		0.007*Stars 
		0.007*Cut Copy 
		0.007*Mumford & Sons 
		0.006*Radiohead 
		0.006*Bon Iver 
		0.006*Hot Chip 
		0.006*MGMT 
		0.006*Phoenix 
		0.006*Kate Nash 
		0.005*Sufjan Stevens 
		0.005*Broken Social Scene

		0.055*2Pac 
		0.041*Jay-Z 
		0.026*The Notorious B.I.G. 
		0.021*Snoop Dogg 
		0.020*Dr. Dre 
		0.016*Kanye West 
		0.015*Lil Wayne 
		0.015*Nas 
		0.014*Young Jeezy 
		0.014*Wu-Tang Clan 
		0.014*Ludacris 
		0.013*Rick Ross 
		0.013*Bone Thugs-n-Harmony 
		0.012*Tupac 
		0.011*The Game 
		0.009*DMX 
		0.009*Nelly 
		0.009*50 Cent 
		0.009*LL Cool J 
		0.008*Fabolous

		0.028*Pearl Jam 
		0.024*Disney 
		0.019*Original Broadway Cast 
		0.019*The Cranberries 
		0.018*The Prodigy 
		0.013*Dream Theater 
		0.013*Enigma 
		0.013*Andrew Lloyd Webber 
		0.013*Slipknot 
		0.012*Scooter 
		0.011*Various Artists 
		0.011*Wicked 
		0.011*Buena Vista Social Club 
		0.009*Compay Segundo 
		0.009*Faithless 
		0.008*Various 
		0.008*Les Miserables 
		0.007*Porcupine Tree 
		0.007*Jonathan Larson 
		0.007*Groove Coverage

		0.044*Tim McGraw 
		0.036*George Strait 
		0.030*Carrie Underwood 
		0.026*Toby Keith 
		0.025*Johnny Cash 
		0.022*Rascal Flatts 
		0.021*Zac Brown Band 
		0.019*Kenny Chesney 
		0.018*Sugarland 
		0.018*Blake Shelton 
		0.018*Josh Turner 
		0.017*Randy Travis 
		0.017*Jason Aldean 
		0.016*Alan Jackson 
		0.015*Brad Paisley 
		0.014*Taylor Swift 
		0.012*Billy Currington 
		0.011*Reba McEntire 
		0.011*Miranda Lambert 
		0.010*Gary Allan

		0.044*Mary J. Blige 
		0.039*Cafe Del Mar 
		0.021*Stéphane Pompougnac 
		0.020*Café Del Mar 
		0.020*A Day to Remember 
		0.019*No te va Gustar 
		0.017*Ratatat 
		0.015*Carlos Santana 
		0.013*Divididos 
		0.012*The KLF 
		0.012*Tyler, The Creator 
		0.011*Tosca 
		0.011*Chicane 
		0.011*Luis Alberto Spinetta 
		0.010*La Vela Puerca 
		0.010*Gaelic Storm 
		0.010*Buddha Bar 
		0.009*Babasónicos 
		0.009*IndieFeed.com Community 
		0.009*Duman

		0.051*Michael Bublé 
		0.045*Frank Sinatra 
		0.024*Maná 
		0.024*Joaquín Sabina 
		0.023*Ricardo Arjona 
		0.020*Luis Miguel 
		0.016*Miranda 
		0.015*Juanes 
		0.014*Alejandro Sanz 
		0.011*Shakira 
		0.009*Alejandro Fernández 
		0.009*Andrés Calamaro 
		0.009*Mana 
		0.008*Rosana 
		0.008*Ricardo Montaner 
		0.008*Andrés Cepeda 
		0.007*La Oreja de Van Gogh 
		0.007*Miguel Bosé 
		0.007*Sin Bandera 
		0.007*Laura Pausini

		0.038*The Black Keys 
		0.028*Drake 
		0.028*Foster The People 
		0.023*Bruno Mars 
		0.020*Florence and the Machine 
		0.020*Rihanna 
		0.020*Taylor Swift 
		0.020*Skrillex 
		0.016*Aaliyah 
		0.015*Red Hot Chili Peppers 
		0.015*Maroon 5 
		0.013*Calvin Harris 
		0.013*Lana Del Rey 
		0.013*Lady Gaga 
		0.012*MGMT 
		0.012*Armin van Buuren 
		0.011*Beyoncé 
		0.011*Avicii 
		0.009*Amy Winehouse 
		0.009*Queen

		0.072*Frédéric Chopin 
		0.055*Johann Sebastian Bach 
		0.050*Wolfgang Amadeus Mozart 
		0.034*Ludwig van Beethoven 
		0.031*Antonio Vivaldi 
		0.024*Beethoven 
		0.019*Chopin 
		0.019*Mozart 
		0.019*Bach 
		0.017*Sting 
		0.016*Claude Debussy 
		0.016*Kenny G        <-- ??!
		0.012*Gal Costa 
		0.011*João Gilberto 
		0.011*Tchaikovsky 
		0.011*Astrud Gilberto 
		0.010*Glenn Gould 
		0.009*Vladimir Horowitz 
		0.009*Caetano Veloso 
		0.008*Georg Friedrich Händel

		0.065*Nine Inch Nails 
		0.027*Slipknot 
		0.021*Rammstein 
		0.021*Ennio Morricone   <-- ??!
		0.019*Daughtry 
		0.019*Rob Zombie 
		0.013*Static-X 
		0.011*Ministry 
		0.011*Children of Bodom 
		0.010*Marilyn Manson 
		0.010*Pet Shop Boys 
		0.010*The Cramps 
		0.010*DevilDriver 
		0.008*Stone Sour 
		0.008*Fairport Convention 
		0.007*White Zombie 
		0.007*Richard O'Brien 
		0.007*Parokya ni Edgar 
		0.007*Ne-Yo 
		0.007*The Brian Setzer Orchestra

		0.043*Jason Mraz 
		0.036*Damian Marley 
		0.031*Relient K 
		0.022*Yiruma 
		0.022*Simple Plan 
		0.021*Casting Crowns 
		0.020*Mavado 
		0.015*Chris Tomlin 
		0.014*Third Day 
		0.013*tobyMac 
		0.013*NEEDTOBREATHE 
		0.012*Vybz Kartel 
		0.012*Stephen Marley 
		0.012*Mägo de Oz 
		0.011*Busy Signal 
		0.011*Hillsong United 
		0.011*MercyMe 
		0.008*BarlowGirl 
		0.008*Buddha-Bar 
		0.008*Goran Bregović

		0.082*Jack Johnson 
		0.051*Stevie Wonder 
		0.031*Ray Charles 
		0.030*Marvin Gaye 
		0.027*Jackson 5 
		0.023*The Temptations 
		0.020*Aretha Franklin 
		0.018*James Brown 
		0.014*Al Green 
		0.013*Otis Redding 
		0.013*Talking Heads 
		0.010*Earth, Wind & Fire 
		0.009*Barry White 
		0.009*Curtis Mayfield 
		0.008*Chuck Berry 
		0.008*The Isley Brothers 
		0.007*Diana Ross & The Supremes 
		0.007*Anthony Hamilton 
		0.006*Cameo 
		0.006*George Benson

		0.098*Skrillex 
		0.095*Sublime 
		0.048*50 Cent 
		0.044*Kirk Franklin 
		0.028*Insane Clown Posse 
		0.020*Amy Winehouse 
		0.013*Jeremy Soule 
		0.013*Fred Hammond 
		0.013*Yolanda Adams 
		0.011*Rod Stewart 
		0.011*Ismael Serrano 
		0.010*Los Aldeanos 
		0.010*Lasse Stefanz 
		0.009*Choking Victim 
		0.008*La Arrolladora Banda El Limon 
		0.008*Jesper Kyd 
		0.008*Israel Houghton 
		0.008*Israel & New Breed 
		0.008*Shekinah Glory Ministry 
		0.008*Bob Marley & The Wailers

		0.047*The Killers 
		0.042*Muse 
		0.035*Red Hot Chili Peppers 
		0.033*U2 
		0.029*The Cure 
		0.025*Jimi Hendrix 
		0.023*Queen 
		0.019*blink-182 
		0.019*The Smiths 
		0.012*Radiohead 
		0.011*Queens of the Stone Age 
		0.011*Stereophonics 
		0.010*Joy Division 
		0.009*The Strokes 
		0.009*Foo Fighters 
		0.008*Keane 
		0.008*Kings of Leon 
		0.007*Led Zeppelin 
		0.007*Stevie Ray Vaughan 
		0.007*Pearl Jam

		0.162*Glee Cast 
		0.130*Various Artists 
		0.031*Glee 
		0.030*Whitney Houston 
		0.017*Boyz II Men 
		0.017*Guns N' Roses 
		0.014*John Legend 
		0.010*David Cook 
		0.010*Tori Amos 
		0.009*Mariah Carey 
		0.007*Jewel 
		0.007*3 Doors Down 
		0.006*Straight No Chaser 
		0.006*Jim Sturgess 
		0.005*Across The Universe 
		0.005*Dru Hill 
		0.005*Jodeci 
		0.005*David Archuleta 
		0.005*Usher 
		0.004*Rhianna

		0.026*Taking Back Sunday 
		0.023*A. R. Rahman 
		0.019*Unknown Artist 
		0.016*New Found Glory 
		0.014*Alkaline Trio 
		0.012*Jimmy Eat World 
		0.012*Ciara 
		0.011*M.I.A. 
		0.009*Mormon Tabernacle Choir 
		0.009*Paul Simon 
		0.008*Saves the Day 
		0.008*Trey Songz 
		0.008*any song in the world 
		0.007*Brand New 
		0.007*Senses Fail 
		0.007*Motion City Soundtrack 
		0.007*Macy Gray 
		0.007*Pulp 
		0.006*Lata Mangeshkar 
		0.006*Thrice

		0.088*Avenged Sevenfold 
		0.068*Tiësto 
		0.024*Nina Simone 
		0.024*Gotan Project 
		0.022*Phish 
		0.022*Bring Me the Horizon 
		0.020*AFI 
		0.019*Carpenters 
		0.016*Paul Oakenfold 
		0.015*Miles Davis 
		0.011*Easy Star All-Stars 
		0.011*Chick Corea 
		0.010*Nicki Minaj 
		0.010*Benny Benassi 
		0.008*Headhunterz 
		0.007*Paul van Dyk 
		0.007*Infected Mushroom 
		0.007*Faith Worship 
		0.007*Five Finger Death Punch 
		0.007*David Benoit

		0.152*Bob Marley 
		0.074*Bob Marley & The Wailers 
		0.062*Enya 
		0.015*Black Uhuru 
		0.014*Bee Gees 
		0.013*Secret Garden 
		0.011*Frank Zappa 
		0.010*Bon Jovi 
		0.009*Immediate Music 
		0.008*Buju Banton 
		0.008*The Four Seasons 
		0.008*Tom Jones 
		0.007*Damian Marley 
		0.007*Chris Isaak 
		0.006*Yoko Shimomura 
		0.006*Vangelis 
		0.006*Barrington Levy 
		0.006*Buddy Holly 
		0.006*Gregory Isaacs 
		0.005*Depeche Mode

		0.033*John Mayer 
		0.022*Mac Miller 
		0.017*Ingrid Michaelson 
		0.016*Mayday Parade 
		0.015*Never Shout Never 
		0.015*Jason Mraz 
		0.015*Silvio Rodríguez 
		0.012*MSTRKRFT 
		0.011*Sara Bareilles 
		0.011*Colbie Caillat 
		0.011*All Time Low 
		0.010*The Maine 
		0.009*Jorge Drexler 
		0.009*Jack Johnson 
		0.008*Secondhand Serenade 
		0.008*Justice 
		0.008*Tokio Hotel 
		0.007*Forever the Sickest Kids 
		0.007*Mercedes Sosa 
		0.007*The Veronicas

		0.077*Nirvana 
		0.034*Soundgarden 
		0.026*Primus 
		0.022*"Weird Al" Yankovic 
		0.022*Stone Temple Pilots 
		0.021*Incubus 
		0.021*Moby 
		0.020*Alice in Chains 
		0.017*Lecrae 
		0.016*311 
		0.015*Molotov 
		0.015*Café Tacvba 
		0.013*The Police 
		0.012*Cafe Tacuba 
		0.011*Sublime 
		0.009*Trip Lee 
		0.009*Soda Stereo 
		0.008*Micheal Jackson 
		0.007*KMFDM 
		0.007*Faith No More

		0.118*The Rolling Stones 
		0.027*Dave Matthews Band 
		0.025*Sublime 
		0.019*Third Eye Blind 
		0.019*Led Zeppelin 
		0.019*Alanis Morissette 
		0.014*Snow Patrol 
		0.012*Jamiroquai 
		0.011*Rage Against the Machine 
		0.011*Garbage 
		0.010*No Doubt 
		0.009*Cat Stevens 
		0.008*Slightly Stoopid 
		0.008*Weezer 
		0.008*Counting Crows 
		0.007*Sheryl Crow 
		0.007*Barenaked Ladies 
		0.007*Putumayo 
		0.007*Tina Dickow 
		0.007*McFly

		0.029*Daft Punk 
		0.027*John Coltrane 
		0.027*Matchbox Twenty 
		0.026*Sigur Rós 
		0.022*Manu Chao 
		0.021*Counting Crows 
		0.019*Lifehouse 
		0.014*B.B. King 
		0.013*Morodo 
		0.013*The Velvet Underground 
		0.010*Tracy Chapman 
		0.010*Gondwana 
		0.009*Bob Dylan 
		0.008*The Goo Goo Dolls 
		0.008*The Fray 
		0.008*James Alan Johnston 
		0.008*Leonard Cohen 
		0.008*Miles Davis 
		0.008*Serge Gainsbourg 
		0.007*Seal

		0.061*Paramore 
		0.056*Death Cab for Cutie 
		0.048*Arctic Monkeys 
		0.045*Backstreet Boys 
		0.042*Fall Out Boy 
		0.041*The White Stripes 
		0.036*My Chemical Romance 
		0.034*Adele 
		0.027*The Shins 
		0.026*Vitamin String Quartet 
		0.022*Tegan and Sara 
		0.021*Dire Straits 
		0.018*Bloc Party 
		0.014*Train 
		0.014*The Kooks 
		0.013*Arcade Fire 
		0.008*Kaiser Chiefs 
		0.008*The Raconteurs 
		0.007*Good Charlotte 
		0.007*Mark Knopfler

		0.029*Guns N' Roses 
		0.016*Def Leppard 
		0.014*The Police 
		0.013*Bryan Adams 
		0.011*Queen 
		0.011*Tears for Fears 
		0.011*Jimmy Cliff 
		0.010*Bon Jovi 
		0.009*Guns Nroses 
		0.009*Ramones 
		0.009*Hall & Oates 
		0.009*U 2 
		0.008*Michael Jackson 
		0.008*Chicago 
		0.008*ABBA 
		0.008*Journey 
		0.008*INXS 
		0.007*Toots & The Maytals 
		0.007*Madonna 
		0.007*Elton John

		0.053*Don Omar 
		0.042*Daddy Yankee 
		0.041*Wisin y Yandel 
		0.039*Aventura 
		0.020*Héctor Lavoe 
		0.019*Marc Anthony 
		0.013*Rakim y Ken-Y 
		0.012*Silvestre Dangond 
		0.011*Gilberto Santa Rosa 
		0.011*Calle 13 
		0.010*Celia Cruz 
		0.010*Carlos Vives 
		0.010*Tego Calderón 
		0.009*Plan B 
		0.009*El Gran Combo de Puerto Rico 
		0.008*Pitbull 
		0.008*Eddie Santiago 
		0.008*Wisin & Yandel 
		0.008*Zion y Lennox 
		0.008*Tony Dize

		0.093*Linkin Park 
		0.055*Naruto 
		0.045*Green Day 
		0.039*These Animal Men 
		0.028*Katy Perry 
		0.027*梶浦由記 
		0.015*Alpha Blondy 
		0.013*Gravitation 
		0.013*Cultura Profética 
		0.010*Los Cafres 
		0.009*Resistencia Suburbana 
		0.008*See-Saw 
		0.008*Bruce Faulconer 
		0.008*Ion Storm 
		0.007*???Y ?R?L 
		0.007*Fruits Basket 
		0.006*You Me At Six 
		0.006*Dread Mar - I 
		0.006*Gundam Wing 
		0.005*Shoji Meguro

		0.056*Rihanna 
		0.045*Hans Zimmer 
		0.042*John Williams 
		0.031*Beyoncé 
		0.030*Howard Shore 
		0.029*James Horner 
		0.020*Evanescence 
		0.020*Britney Spears 
		0.020*Loreena McKennitt 
		0.018*Alexandre Desplat 
		0.014*Lil Boosie 
		0.012*Christina Aguilera 
		0.012*Miley Cyrus 
		0.011*Ion Storm 
		0.010*The Pussycat Dolls 
		0.010*Mariah Carey 
		0.010*Fergie 
		0.009*Steve Jablonsky 
		0.009*Infected Mushroom 
		0.008*James Newton Howard

		0.263*Lil Wayne 
		0.029*Trey Songz 
		0.026*Lady Gaga 
		0.024*Madonna 
		0.016*Nicki Minaj 
		0.015*Pitbull 
		0.011*LMFAO 
		0.011*Kanye West 
		0.011*Robin Thicke 
		0.010*Soulja Boy Tell 'Em 
		0.010*David Guetta 
		0.009*Akon 
		0.009*Chris Brown 
		0.008*Rihanna 
		0.008*Kevin Rudolf 
		0.008*Usher 
		0.007*Paolo Nutini 
		0.007*Pretty Ricky 
		0.007*Ne-Yo 
		0.006*Trina

		0.040*Creedence Clearwater Revival 
		0.039*Elton John 
		0.037*The Who 
		0.035*Fleetwood Mac 
		0.021*Steve Miller Band 
		0.018*Lynyrd Skynyrd 
		0.017*Led Zeppelin 
		0.016*The Rolling Stones 
		0.014*Rush 
		0.014*Andrés Calamaro 
		0.013*The Doors 
		0.013*Steely Dan 
		0.012*Eagles 
		0.012*Eric Clapton 
		0.011*Bersuit Vergarabat 
		0.009*Styx 
		0.009*Patricio Rey y sus Redonditos de Ricota 
		0.009*Kid Rock 
		0.009*Simon & Garfunkel 
		0.008*Supertramp

		0.025*Massive Attack 
		0.019*Pendulum 
		0.018*Weezer 
		0.018*Girl Talk 
		0.017*Mt Eden 
		0.012*Portishead 
		0.011*소녀시대 
		0.009*Skrillex 
		0.009*DJ Shadow 
		0.008*Nujabes 
		0.008*Bonobo 
		0.008*Morcheeba 
		0.008*Flux Pavilion 
		0.007*Amon Tobin 
		0.007*RJD2 
		0.007*Kanye West 
		0.007*The Glitch Mob 
		0.007*2NE1 
		0.007*Stephen Lynch 
		0.007*Lamb

		0.077*Maroon 5 
		0.045*Skillet 
		0.037*Hollywood Undead 
		0.032*Akon 
		0.031*T.I. 
		0.021*Young Jeezy 
		0.021*Panic! At the Disco 
		0.020*Josh Groban 
		0.020*blink-182 
		0.020*Rick Ross 
		0.017*Plies 
		0.017*Sum 41 
		0.015*Ne-Yo 
		0.014*Rancid 
		0.013*T-Pain 
		0.013*The Misfits 
		0.012*Ed Sheeran  
		0.009*Craig David 
		0.009*NOFX 
		0.008*Bad Religion

		0.031*Eric Clapton 
		0.026*Neil Young 
		0.024*Fleet Foxes 
		0.021*Bassnectar 
		0.021*Ion Storm 
		0.019*Ray LaMontagne 
		0.019*Boyce Avenue 
		0.015*The Avett Brothers 
		0.014*The Strokes 
		0.013*The Doors 
		0.010*Brand New 
		0.009*John Lee Hooker 
		0.008*Eddie Vedder 
		0.008*Bruce Springsteen 
		0.008*The Black Keys 
		0.008*My Morning Jacket 
		0.007*Guster 
		0.007*Dr. Dog 
		0.007*Balkan Beat Box 
		0.006*Angels & Airwaves

		0.051*Atmosphere 
		0.049*Jonas Brothers           <-- ???
		0.038*Thievery Corporation 
		0.036*Björk 
		0.033*Marilyn Manson 
		0.030*Zero 7 
		0.029*林俊傑 
		0.028*Grateful Dead 
		0.026*John Zorn 
		0.026*王力宏 
		0.026*Jay Chou 周杰伦 
		0.022*五月天 
		0.013*陶喆 
		0.012*Adam Lambert 
		0.012*Chris Rice 
		0.012*張學友 
		0.011*羅志祥 
		0.011*周杰倫 (Jay Chou) 
		0.010*Cam Ly 
		0.010*Kings of Leon


The topics are not perfect, some strange occurrences can be spotted here and there. Let's see if we can get better results using Mallet. 

#### Mallet 

After few tests only I settled for this configuration:
* 150 topics.
* alpha = 0.005
* beta = 0.5
* we are using the large dataset here: 225413 playlists and 18729 artists. 

The topics extracted are shown at the end of the README. I can make sense of those topics more easily than those extracted using Gensim. We consider the same query as for LSI :
 
		artist name              count
		Lynyrd Skynyrd :           1
		Black Sabbath :            5
		Metallica :                5
		Iron Maiden :              3
		Jimi Hendrix :             5
		John Zorn :                3

We can get the topic distribution theta of this document in few step of gibbs sampling. Once we have theta, and with the phi obtained during the training phase, we can compute a score for each artist. This score correspond to the probability of that artist to be generated knowing theta and phi:

score(w) = sum_{topic z} (theta[z]*phi[w,z])

We recommend artists with high scores:

		Tool  0.052073927209028766
		Pantera  0.047459466254608965
		Megadeth  0.026386380633378392
		Slayer  0.022075302591185862
		Creedence Clearwater Revival  0.0209437763460108
		Radiohead  0.019508512090458077
		The Rolling Stones  0.018207798325406494
		Rage Against the Machine  0.017578908027905262
		Sepultura  0.016742082016805437
		John Williams  0.016001947820583017
		Death Cab for Cutie  0.015927669208999778
		Hans Zimmer  0.015448013660030227
		Motörhead  0.014956960624116908
		Alice in Chains  0.011270160175612684
		Dream Theater  0.011268369317160541
		Led Zeppelin  0.010539134556468325
		Howard Shore  0.009967907955768747
		Dethklok  0.009939791967223546
		Creed  0.009925818053173314
		The Who  0.009684184628956453
		Deftones  0.009318918987887488
		Anthrax  0.008819561323790777
		Ion Storm  0.008618885344008793
		The Doors  0.008244804687153775
		Fear Factory  0.008093485660269992

#### Serendipity

Just as with LSI we might lack of serendipity here. When working with LDA it is possible to play on the hyperparameter alpha to increase or decrease the sparsity of the topics mixtures. This can be a way to leverage serendipity. If we increase alpha then a playlist will be considered as a mixture of many topics and artists that may be not obviously related to the playlist might appear in the recommendation. This however might decrease the precision of the system.

#### Topics obtained using Mallet

		0
		Owl City (9835)
		Steve Miller Band (1804)
		3OH!3 (1768)
		Eminem (1502)
		Ke$ha (1287)
		Kid Cudi (1082)
		Drake (1047)
		Timbaland (920)
		Glee Cast (907)
		Hinder (847)

		1
		Adele (4803)
		Mogwai (1331)
		Va - www.musicasparabaixar.org (1331)
		Moulin Rouge (1227)
		Cannibal Corpse (844)
		Brandi Carlile (701)
		Craig Armstrong (431)
		Olly Murs (374)
		Hundred Waters (319)
		Cocoon (287)

		2
		James Blunt (3959)
		Take That (768)
		Hocus Pocus (608)
		Blumentopf (598)
		Murs (572)
		MC Solaar (498)
		Prinz Pi (428)
		Sido (410)
		Medeski Martin and Wood (346)
		Kool Savas (338)

		3
		Ella Fitzgerald (7032)
		Ray Charles (5860)
		Louis Armstrong (5009)
		Diana Krall (4959)
		Nina Simone (4485)
		Etta James (3468)
		Billie Holiday (3245)
		Primus (1696)
		Madeleine Peyroux (1629)
		Sarah Vaughan (1497)

		4
		Jamiroquai (3462)
		James Brown (2530)
		Earth, Wind & Fire (2089)
		Kool & The Gang (1615)
		Funkadelic (1310)
		Parliament (1297)
		Cameo (1006)
		Fela Kuti (1003)
		Chaka Khan (930)
		Curtis Mayfield (794)

		5
		Mercedes Sosa (2941)
		Elliott Smith (2475)
		LMFAO (2198)
		Coheed and Cambria (1612)
		Rita Lee (1212)
		John Pizzarelli (882)
		Brazilian Tropical Orchestra (742)
		Richie Havens (714)
		Gwen Stefani (651)
		Peteco Carabajal (645)

		6
		Elton John (8773)
		Bee Gees (6225)
		Girl Talk (5119)
		Chicago (2711)
		Bread (2613)
		The Mamas & the Papas (1860)
		James Taylor (1780)
		Carpenters (1779)
		Simon & Garfunkel (1687)
		Peter, Paul & Mary (1473)

		7
		Tim McGraw (7661)
		Lecrae (3638)
		Trip Lee (1701)
		Shania Twain (1363)
		Tedashii (798)
		Flame (701)
		116 Clique (687)
		Pro (538)
		Sho Baraka (535)
		The Blues Collection (502)

		8
		Muse (20147)
		The Shins (4069)
		Keane (3369)
		Bad Religion (2757)
		Angels & Airwaves (1344)
		Spoon (545)
		Diego el Cigala (540)
		Grandaddy (448)
		Pixies (319)
		The Rasmus (311)

		9
		Nickelback (5255)
		Never Shout Never (4518)
		Mayday Parade (3666)
		Simple Plan (3375)
		Daughtry (3050)
		The All-American Rejects (2907)
		Secondhand Serenade (2900)
		All Time Low (2535)
		The Maine (2291)
		Lifehouse (1997)

		10
		Yeah Yeah Yeahs (6452)
		The White Stripes (5937)
		Bloc Party (3313)
		Arctic Monkeys (2052)
		The Kooks (1900)
		The Strokes (1780)
		Kaiser Chiefs (1720)
		Kasabian (1716)
		The Hives (1589)
		The Raconteurs (1533)

		11
		Disturbed (5580)
		Papa Roach (3587)
		Ismael Serrano (2722)
		Swing Out Sister (656)
		Jorge Drexler (510)
		Three Days Grace (473)
		Gerardo Ortiz (470)
		Los Cadetes de Linares (451)
		D'Angelo (438)
		Big Time Rush (357)

		12
		Ingrid Michaelson (7492)
		Colbie Caillat (6215)
		Sara Bareilles (3662)
		Jason Mraz (1883)
		Matt Nathanson (1765)
		Joshua Radin (1642)
		Regina Spektor (1328)
		Missy Higgins (1316)
		Matt Wertz (1229)
		John Mayer (1202)

		13
		Dropkick Murphys (3158)
		Streetlight Manifesto (2376)
		Mormon Tabernacle Choir (2048)
		Less Than Jake (1851)
		NOFX (1826)
		The Mighty Mighty Bosstones (1795)
		Millencolin (1613)
		Reel Big Fish (1554)
		Me First and the Gimme Gimmes (1477)
		Against Me! (1070)

		14
		Maná (6603)
		Ricardo Arjona (6561)
		Luis Miguel (5800)
		Juanes (3925)
		Alejandro Sanz (3799)
		Shakira (3006)
		Laura Pausini (2747)
		Mana (2370)
		Miguel Bosé (2357)
		Ricardo Montaner (2095)

		15
		The Black Keys (11447)
		Mumford & Sons (11170)
		Florence and The Machine (9905)
		Florence and the Machine (4282)
		Cold War Kids (2784)
		Coldplay (1760)
		Young the Giant (1578)
		Modest Mouse (1405)
		Edward Sharpe & The Magnetic Zeros (1369)
		Arcade Fire (1327)

		16
		Maroon 5 (16055)
		Regina Spektor (6873)
		Snow Patrol (4882)
		The Fray (4086)
		The Cranberries (3533)
		OneRepublic (3111)
		Train (2924)
		The Script (2045)
		Lifehouse (1228)
		Gavin DeGraw (1177)

		17
		Limp Bizkit (8445)
		Tech N9ne (7740)
		Kid Rock (2983)
		Insane Clown Posse (2676)
		South Park Mexican (2553)
		Cypress Hill (1964)
		Mac Dre (1608)
		Three 6 Mafia (1302)
		311 (1253)
		Afroman (1024)

		18
		2Pac (20098)
		Jay-Z (14574)
		The Notorious B.I.G. (8777)
		Tupac (5175)
		Bone Thugs-n-Harmony (4127)
		Fabolous (2894)
		The Game (2235)
		Tupac Shakur (1887)
		Nas (1356)
		50 Cent (1009)

		19
		Nightwish (6331)
		Kamelot (3285)
		Helloween (3223)
		Sonata Arctica (3102)
		Manowar (3034)
		DragonForce (2895)
		Iron Maiden (2618)
		Moby (2554)
		Stratovarius (1947)
		Rhapsody (1669)

		20
		植松伸夫 (5350)
		Yoko Shimomura (2480)
		近藤浩治 (2111)
		Jeremy Soule (2055)
		Kingdom Hearts (1970)
		Yasunori Mitsuda (1693)
		Zelda (1638)
		Nintendo (1589)
		Final Fantasy (1287)
		Martin O'Donnell & Michael Salvatori (1277)

		21
		Lil Boosie (5841)
		Flight of the Conchords (2171)
		Missy Elliott (1695)
		Tenacious D (1598)
		Mägo de Oz (1518)
		Webbie (1097)
		Trina (819)
		Pretty Ricky (637)
		Avalanch (544)
		Nicola Conte (510)

		22
		James Horner (2716)
		소녀시대 (2657)
		2NE1 (1986)
		Big Bang (1622)
		Super Junior (1608)
		Epik High (1240)
		BigBang (1155)
		2AM (1018)
		Big Bang (1000)
		IU (992)

		23
		Faith No More (5847)
		Wiz Khalifa (4481)
		REM (906)
		Benny Benassi (870)
		Buika (746)
		Poets of the Fall (586)
		Camarón de la Isla (496)
		Franz Schubert (402)
		Bebo Valdes & Javier Colina (382)
		Mr. Bungle (381)

		24
		Carrie Underwood (9083)
		Tim McGraw (8293)
		Rascal Flatts (7733)
		Toby Keith (7672)
		Zac Brown Band (7294)
		Sugarland (6588)
		Kenny Chesney (6206)
		Jason Aldean (5966)
		Blake Shelton (5940)
		Brad Paisley (4492)

		25
		Andrew Bird (4474)
		Beck (3036)
		Frank Zappa (2742)
		Beirut (2561)
		Frightened Rabbit (1520)
		Rilo Kiley (1219)
		Joanna Newsom (1196)
		Animal Collective (1193)
		Man Man (1171)
		Devendra Banhart (1080)

		26
		My Chemical Romance (3912)
		Brand New (3475)
		Mindless Self Indulgence (3060)
		AFI (2601)
		The Used (2217)
		HIM (1662)
		Thrice (1472)
		Lostprophets (1279)
		Story of the Year (1245)
		Bayside (1146)

		27
		Ray LaMontagne (5641)
		Katy Perry (3641)
		Tegan and Sara (3195)
		Regina Spektor (1797)
		Sia (1384)
		Tegan And Sara (908)
		emiliana torrini (744)
		Feist (721)
		Imogen Heap (694)
		Kruder & Dorfmeister (610)

		28
		Damian Marley (6682)
		Mavado (3483)
		Vybz Kartel (2178)
		Busy Signal (2095)
		Stephen Marley (2095)
		Buju Banton (1586)
		Elephant Man (1425)
		Sean Paul (1260)
		Pitbull (1134)
		Lady Saw (1091)

		29
		Manu Chao (8432)
		Buena Vista Social Club (5313)
		Héctor Lavoe (3156)
		El Gran Combo de Puerto Rico (1874)
		Fania All-Stars (1462)
		Joe Arroyo (1286)
		Celia Cruz (1134)
		Willie Colón (1124)
		The Cat Empire (1094)
		Ismael Rivera (976)

		30
		Enya (12641)
		Skrillex (10984)
		Skillet (8250)
		deadmau5 (7577)
		Pillar (758)
		Disciple (714)
		Imogen Heap (683)
		Alex Gaudino (444)
		Far East Movement (421)
		Axwell (419)

		31
		Jack Johnson (28885)
		Damien Rice (2288)
		Tracy Chapman (1975)
		Calvin Harris (1695)
		David Gray (1618)
		Xavier Rudd (1086)
		Donavon Frankenreiter (946)
		Newton Faulkner (653)
		G. Love & Special Sauce (603)
		Alexi Murdoch (571)

		32
		Benga (5371)
		Skream (4849)
		Vitamin String Quartet (4540)
		Bar 9 (4058)
		Rusko (3758)
		La Roux (3539)
		DJ /rupture (2838)
		Nero (2246)
		Datsik (2126)
		Mt Eden (1871)

		33
		Drake (6812)
		Rihanna (5026)
		Eminem (4980)
		Foster The People (4888)
		Bruno Mars (4173)
		The Black Keys (4080)
		Red Hot Chili Peppers (3475)
		Skrillex (3448)
		Bob Marley (3205)
		Lana Del Rey (2934)

		34
		Cafe Del Mar (3587)
		Gotan Project (2833)
		Café Del Mar (2324)
		Stéphane Pompougnac (1950)
		Tosca (1523)
		St. Germain (1427)
		Chicane (1188)
		Thievery Corporation (1183)
		Various Artists (1083)
		Groove Armada (995)

		35
		Hillsong (2638)
		Jesus Adrian Romero (1434)
		Hillsong United (1167)
		Rodrigo (1110)
		Damas Gratis (790)
		La champions liga (774)
		Faith Worship (770)
		Karina (763)
		El Original (753)
		La Nueva Luna (702)

		36
		Justin Bieber (27204)
		Selena Gomez (4888)
		Miley Cyrus (4342)
		Demi Lovato (2839)
		Taylor Swift (2770)
		Selena Gomez & The Scene (1975)
		Big Time Rush (1052)
		Cody Simpson (1051)
		Beyoncé (1025)
		Victoria Justice (1006)

		37
		Timbaland (2049)
		Reverend Horton Heat (1383)
		Craig David (1260)
		Justin Timberlake (787)
		The Cramps (706)
		The Meteors (691)
		Putumayo World Music (656)
		Gene Vincent (633)
		Stray Cats (579)
		Robin Thicke (522)

		38
		Guns N' Roses (17287)
		Aerosmith (4187)
		Guns Nroses (1716)
		John Mellencamp (1025)
		Bryan Adams (787)
		31 Minutos (738)
		The Cure (619)
		Aero Smith (539)
		Best Of '80 (358)
		The Tubes (342)

		39
		Kid Cudi (18826)
		AC/DC (9625)
		Mac Miller (2413)
		Childish Gambino (2230)
		Ray Conniff (1022)
		Shwayze (985)
		Chiddy Bang (946)
		Era (908)
		Sam Adams (908)
		Lupe Fiasco (807)

		40
		Fleet Foxes (6068)
		The Avett Brothers (4614)
		The Tallest Man on Earth (2206)
		Wilco (2190)
		Noah and the Whale (2170)
		Paolo Nutini (2163)
		Mumford & Sons (1841)
		Angus & Julia Stone (1667)
		James Morrison (1408)
		Bon Iver (1388)

		41
		Bon Jovi (9397)
		A. R. Rahman (4715)
		Toto (1335)
		Jagjit Singh (960)
		CunninLynguists (885)
		सोनू निगम (880)
		Udit Narayan (856)
		Kishore Kumar (852)
		Big Bad Voodoo Daddy (820)
		Lata Mangeshkar (817)

		42
		The Police (2452)
		Rod Stewart (1806)
		Lucinda Williams (1416)
		John Prine (1241)
		Patty Griffin (1217)
		Lyle Lovett (990)
		Elvis Costello (945)
		John Hiatt (848)
		Emmylou Harris (682)
		Ryan Adams (666)

		43
		Die Ärzte (2087)
		ABBA (1794)
		La Renga (1570)
		Clueso (756)
		Richard Cheese (734)
		Beatsteaks (671)
		Seal (633)
		Seeed (606)
		Fairport Convention (568)
		Lasse Stefanz (564)

		44
		Beach House (2581)
		James Blake (1801)
		TV on the Radio (1653)
		toro y moi (1553)
		Grizzly Bear (1300)
		The Flaming Lips (1292)
		Tame Impala (1060)
		Best Coast (951)
		Animal Collective (819)
		Baths (806)

		45
		Daft Punk (6094)
		MSTRKRFT (3482)
		Cut Copy (3415)
		Justice (3132)
		Crystal Castles (3018)
		Josh Groban (2804)
		Hot Chip (2370)
		Boys Noize (2037)
		MGMT (1537)
		Booka Shade (1465)

		46
		Nine Inch Nails (11725)
		Tom Waits (6473)
		Ministry (2623)
		KMFDM (1448)
		Tool (1306)
		Pixies (1140)
		Sonic Youth (1132)
		Hole (863)
		My Life With the Thrill Kill Kult (847)
		Type O Negative (793)

		47
		0.690
		Atmosphere (7745)
		Immortal Technique (3452)
		Nujabes (3362)
		Jedi Mind Tricks (3237)
		Jurassic 5 (3205)
		Aesop Rock (2638)
		Sage Francis (2073)
		MF DOOM (1592)
		Brother Ali (1137)
		Deltron 3030 (1014)

		48
		五月天 (2533)
		林俊傑 (2486)
		王力宏 (1524)
		周杰倫 (1304)
		소녀시대 (1235)
		羅志祥 (1045)
		陳奕迅 (1040)
		周杰倫 (Jay Chou) (959)
		梁靜茹 (870)
		蔡依林 (798)

		49
		George Strait (10574)
		Randy Travis (2901)
		Alan Jackson (2814)
		Reba McEntire (2683)
		Cross Canadian Ragweed (2252)
		Randy Rogers Band (1873)
		Tim McGraw (1805)
		Toby Keith (1757)
		Travis Tritt (1667)
		Gary Allan (1639)

		50
		Cat Stevens (3629)
		Steely Dan (3551)
		Tom Petty (3550)
		Supertramp (3348)
		Tom Petty and The Heartbreakers (3233)
		Styx (3050)
		Eagles (2583)
		Bruce Springsteen (2217)
		Fleetwood Mac (2127)
		38 Special (2000)

		51
		Silvio Rodríguez (7766)
		João Gilberto (2822)
		Ludovico Einaudi (2821)
		Antônio Carlos Jobim (2717)
		Astrud Gilberto (2687)
		Tyler, The Creator (1888)
		Sérgio Mendes (1665)
		Pablo Milanes (1405)
		Monster Demolition Night (1080)
		Compay Segundo (987)

		52
		Parokya ni Edgar (1377)
		Ministério de Louvor Diante do Trono (1240)
		Gipsy Kings (1206)
		Gogol Bordello (1071)
		Paco de Lucía (1021)
		M.Y.M.P. (919)
		Fernandinho (692)
		Gary Valenciano (601)
		Sarah Geronimo (574)
		Oficina G3 (560)

		53
		The Allman Brothers Band (1769)
		Willie Nelson (1726)
		Hank Williams (1721)
		George Jones (1608)
		Hank Williams, Jr. (1568)
		Patsy Cline (1544)
		Drive-By Truckers (1218)
		Amon Tobin (1121)
		Johnny Cash (1042)
		Merle Haggard (951)

		54
		Loreena McKennitt (6512)
		Thievery Corporation (4046)
		Queens of the Stone Age (3521)
		Enya (2321)
		Secret Garden (2234)
		Dead Can Dance (1995)
		Clannad (1694)
		Ratatat (1683)
		喜多郎 (919)
		Vangelis (880)

		55
		0.001
		Lupe Fiasco (13911)
		Rihanna (13894)
		Chris Brown (12205)
		Eminem (10375)
		Lil Wayne (9868)
		Flo Rida (9058)
		Kanye West (8441)
		B.o.B (8440)
		The Black Eyed Peas (7654)
		The Lonely Island (6373)

		56
		Jonas Brothers (9151)
		Miley Cyrus (2352)
		Kenny G (2213)
		Tokio Hotel (2047)
		The Veronicas (1966)
		N-Dubz (1711)
		Avril Lavigne (1654)
		Kanye West (1469)
		Ashley Tisdale (1075)
		Skepta (1070)

		57
		Snoop Dogg (12094)
		Dr. Dre (8538)
		DMX (5423)
		Wu-Tang Clan (4158)
		The Notorious B.I.G. (2915)
		Nate Dogg (2806)
		Busta Rhymes (2601)
		Method Man (2401)
		Ice Cube (2268)
		Jay-Z (2229)

		58
		Parov Stelar (3330)
		Bajofondo (2109)
		Chinese Man (1325)
		Apparat (1272)
		Gramatik (1208)
		Nicolas Jaar (1178)
		Caravan Palace (1168)
		Trentemøller (1120)
		Wax Tailor (1072)
		James Alan Johnston (1012)

		59
		Rihanna (20232)
		Madonna (8047)
		Britney Spears (7506)
		Pink (6917)
		Christina Aguilera (3461)
		Beyoncé (3298)
		The Pussycat Dolls (3044)
		Lady Gaga (2388)
		Justin Timberlake (1940)
		Nelly Furtado (1884)

		60
		The Black Eyed Peas (10624)
		Taio Cruz (1831)
		The Corrs (1014)
		Michael Franks (665)
		Suede (492)
		Kenny Rankin (473)
		George Benson (450)
		Jorge Aragão (390)
		Zeca Pagodinho (380)
		The Manhattan Transfer (348)

		61
		Creedence Clearwater Revival (14784)
		The Rolling Stones (12796)
		Led Zeppelin (7405)
		Jimi Hendrix (7345)
		Lynyrd Skynyrd (6919)
		The Who (6819)
		The Doors (5804)
		AC/DC (4384)
		ZZ Top (2906)
		The Police (2480)

		62
		B.B. King (3199)
		John Lee Hooker (2688)
		Blues Brothers (2109)
		Stevie Ray Vaughan (2102)
		Muddy Waters (2005)
		Joe Bonamassa (1406)
		Albert King (1134)
		Howlin' Wolf (1094)
		Blues Traveler (1043)
		2 Unlimited (999)

		63
		Two Door Cinema Club (5396)
		Vampire Weekend (5029)
		Passion Pit (4299)
		MGMT (3803)
		Foster The People (2407)
		The Naked And Famous (2271)
		Phoenix (1886)
		Miike Snow (1845)
		Florence and The Machine (1821)
		Yeah Yeah Yeahs (1776)

		64
		Marc Anthony (2350)
		Pet Shop Boys (1698)
		Silvestre Dangond (1695)
		La Arrolladora Banda El Limon (1694)
		Gilberto Santa Rosa (1454)
		Pedro Infante (1372)
		Eddie Santiago (1369)
		Banda El Recodo (1367)
		Celia Cruz (1299)
		Joan Sebastian (1280)

		65
		0.021
		Van Halen (5916)
		Scorpions (5167)
		KISS (5167)
		Def Leppard (4455)
		AC/DC (2562)
		Elton John (2181)
		Tesla (2034)
		Skid Row (1665)
		Guns N' Roses (1481)
		Cinderella (1459)

		66
		Extremoduro (2304)
		Marea (1435)
		Fito & Fitipaldis (1192)
		Headhunterz (978)
		Reincidentes (901)
		Platero y Tu (857)
		La Fuga (829)
		Barricada (789)
		Manel (741)
		La Polla Records (702)

		67
		The Doors (6132)
		The Prodigy (3989)
		The Crystal Method (1947)
		Zaz (1147)
		Pavement (767)
		John Frusciante (677)
		Lou Gramm (568)
		Candlebox (556)
		Silverchair (513)
		Mashup-Germany (508)

		68
		Amy Winehouse (8350)
		Portishead (3972)
		Tricky (2464)
		Duffy (2340)
		Janis Joplin (1366)
		The Lemonheads (1248)
		Kimya Dawson (981)
		Blondie (952)
		Frank Ocean (945)
		Dr Demento (943)

		69
		The Offspring (6152)
		AC/DC (4388)
		Metallica (3728)
		Bloodhound Gang (3231)
		Red Hot Chili Peppers (3017)
		Billy Talent (2802)
		Foo Fighters (2433)
		Muse (2160)
		Nirvana (1988)
		Linkin Park (1821)

		70
		U2 (15996)
		The Cure (13078)
		The Smiths (7891)
		Dire Straits (4373)
		Joy Division (3513)
		New Order (2425)
		Siouxsie and the Banshees (1583)
		Mark Knopfler (1249)
		Cradle of Filth (1147)
		R.E.M. (1017)

		71
		Explosions in the Sky (3742)
		Rise Against (3742)
		Philip Glass (2717)
		The Album Leaf (1674)
		The Postal Service (1664)
		Alexandre Desplat (1328)
		Silversun Pickups (1302)
		Godspeed You! Black Emperor (1010)
		Boards of Canada (978)
		Telefon Tel Aviv (937)

		72
		Los Cafres (4091)
		Gondwana (3894)
		Cultura Profética (3509)
		SOJA (2372)
		Morodo (2173)
		Nonpalidece (1921)
		Dread Mar - I (1788)
		Resistencia Suburbana (1311)
		any song in the world (1002)
		Choc Quib Town (814)

		73
		Avenged Sevenfold (12546)
		Hollywood Undead (8561)
		Bullet for My Valentine (3524)
		Atreyu (2502)
		All That Remains (2063)
		Escape the Fate (1500)
		Stone Sour (1069)
		Drowning Pool (888)
		Black Veil Brides (626)
		ZZ Top (604)

		74
		Lil Wayne (79486)
		Gucci Mane (8792)
		T.I. (6259)
		Young Jeezy (3924)
		Waka Flocka Flame (3046)
		The Game (1774)
		Kevin Rudolf (1331)
		Birdman (1248)
		Yo Gotti (1056)
		Gorilla Zoe (735)

		75
		Stevie Wonder (14473)
		Marvin Gaye (7341)
		The Temptations (5382)
		Aretha Franklin (3925)
		Otis Redding (3312)
		Jackson 5 (3243)
		Al Green (2724)
		Barry White (2691)
		Ray Charles (2172)
		James Brown (1762)

		76
		Cafe Tacuba (3685)
		Soda Stereo (3432)
		Johannes Brahms (2814)
		Molotov (2769)
		Café Tacvba (2645)
		Los Fabulosos Cadillacs (2310)
		Gustavo Cerati (1724)
		Calle 13 (1683)
		Zoé (1607)
		La Ley (1370)

		77
		Rihanna (7316)
		Jessie J (4122)
		The Black Eyed Peas (3959)
		Nicki Minaj (3729)
		David Guetta (3694)
		Lady Gaga (3535)
		Flo Rida (3529)
		Bruno Mars (2973)
		LMFAO (2870)
		Maroon 5 (2869)

		78
		Queen (9711)
		U 2 (2471)
		Scissor Sisters (1763)
		Journey (1752)
		Ladytron (1467)
		James (1228)
		Michael Jackson (1161)
		Asia (1158)
		Roxette (1024)
		Guns Nroses (963)

		79
		Techno (4395)
		Armin van Buuren (3709)
		Tiësto (3513)
		Paul Oakenfold (2836)
		Scooter (2246)
		Cascada (2117)
		Gigi D'Agostino (1931)
		Basshunter (1882)
		Paul van Dyk (1646)
		BT (1384)

		80
		Casting Crowns (3725)
		Chris Tomlin (2997)
		MercyMe (2844)
		tobyMac (2351)
		NEEDTOBREATHE (2119)
		Third Day (1969)
		Chris Rice (1773)
		Michael W. Smith (1633)
		Hillsong United (1512)
		BarlowGirl (1450)

		81
		John Mayer (8509)
		The Mars Volta (1372)
		Paul Kalkbrenner (1331)
		Charly García (1152)
		INXS (1046)
		The Weeknd (927)
		Angelo Badalamenti (744)
		The Flashbulb (572)
		tan bionica (306)
		Hilltop Hoods (289)

		82
		Wisin y Yandel (9118)
		Don Omar (8653)
		Daddy Yankee (8430)
		Aventura (6397)
		Pitbull (5731)
		Stereophonics (2204)
		Rakim y Ken-Y (1973)
		Plan B (1836)
		Wisin & Yandel (1675)
		Tego Calderón (1563)

		83
		Led Zeppelin (11593)
		Johnny Cash (9280)
		Eva Cassidy (1913)
		Gypsy Kings (677)
		Robert Plant (596)
		Billy Ocean (564)
		AC/DC (523)
		Deep Purple (392)
		Ray Stevens (391)
		Eric Johnson (285)

		84
		Sting (6250)
		Yiruma (3222)
		Boney James (1550)
		David Benoit (1111)
		Incognito (1080)
		Carlos Santana (1041)
		Twista (1012)
		The Police (819)
		Norman Brown (737)
		Chris Botti (698)

		85
		Slipknot (10363)
		Marilyn Manson (7894)
		Rob Zombie (4838)
		System of a Down (4680)
		System of a Down (3760)
		Rammstein (3595)
		Trivium (2132)
		Static-X (1984)
		Apocalyptica (1955)
		Deftones (1797)

		86
		Nicki Minaj (13052)
		Trey Songz (12219)
		Chris Brown (11102)
		Usher (9058)
		Beyoncé (6874)
		Ne-Yo (6174)
		Keri Hilson (2972)
		Rihanna (2734)
		Ashanti (2385)
		Ciara (2293)

		87
		Hall & Oates (3298)
		Tears for Fears (3261)
		The Police (2490)
		Erasure (1922)
		The Cure (1849)
		Depeche Mode (1519)
		INXS (1289)
		Various Artists (1233)
		Aha (1229)
		Madonna (1204)

		88
		Metric (2927)
		Paul Simon (2307)
		Sufjan Stevens (2021)
		Rockapella (1667)
		Straight No Chaser (1532)
		Lila Downs (731)
		Kill Bill (632)
		OST (601)
		Lhasa (536)
		Melendi (484)

		89
		Bassnectar (7840)
		Pretty Lights (3279)
		Beats Antique (1523)
		Girl Talk (1309)
		Astor Piazzolla (1074)
		Steve Earle (930)
		STS9 (835)
		The Hood Internet (751)
		Julio Sosa (680)
		Gotan Project (611)

		90
		0.001
		U2 (4812)
		R.E.M. (2891)
		Elton John (2372)
		Sting (2341)
		Various Artists (2228)
		The Rolling Stones (2139)
		The Cardigans (2015)
		Michael Jackson (1999)
		Queen (1854)
		Phil Collins (1852)

		91
		Akon (11483)
		Lil Wayne (9543)
		Ludacris (7756)
		Plies (6601)
		T-Pain (5623)
		Nelly (5590)
		T.I. (5466)
		50 Cent (5107)
		Young Jeezy (3908)
		Kanye West (3863)

		92
		Paramore (9934)
		Panic! At the Disco (5992)
		Relient K (4053)
		Dashboard Confessional (4012)
		New Found Glory (3080)
		Taking Back Sunday (2876)
		Jimmy Eat World (2588)
		Alkaline Trio (1796)
		Motion City Soundtrack (1519)
		Something Corporate (1506)

		93
		Alanis Morissette (6400)
		Backstreet Boys (5494)
		*NSync (3398)
		Bryan Adams (3072)
		Westlife (1928)
		Lionel Richie (1886)
		Jewel (1857)
		Sheryl Crow (1759)
		Savage Garden (1681)
		Kelly Clarkson (1663)

		94
		Eminem (20773)
		Justin Bieber (20452)
		Florence and The Machine (13881)
		Mumford & Sons (10011)
		The Black Eyed Peas (9638)
		Kid Cudi (8293)
		Lil Wayne (7868)
		Chris Brown (6343)
		Enrique Iglesias (5927)
		Bob Marley (5627)

		95
		Glee Cast (37762)
		Various Artists (19393)
		Glee (9579)
		Original Broadway Cast (3582)
		Andrew Lloyd Webber (2460)
		Wicked (1802)
		Les Miserables (1506)
		Jonathan Larson (1382)
		Rent Mel (1178)
		Les Misérables Original Broadway Cast (775)

		96
		0.272
		The Roots (4860)
		Talib Kweli (2785)
		John Legend (2606)
		Mos Def (2591)
		Common (2559)
		Nas (2517)
		OutKast (2022)
		Kanye West (2021)
		De La Soul (1645)
		A Tribe Called Quest (1568)

		97
		Drake (33134)
		Rick Ross (7416)
		Mac Miller (5751)
		Tyga (4664)
		J. Cole (4575)
		Wiz Khalifa (4070)
		Wale (3353)
		Lil Wayne (2800)
		Big Sean (2651)
		Chris Brown (2365)

		98
		The Killers (17828)
		Red Hot Chili Peppers (17509)
		Kings of Leon (3420)
		Red Hot Chilli Peppers (897)
		Brandon Flowers (678)
		Seether (489)
		Belle and Sebastian (409)
		The Strokes (396)
		Amália Rodrigues (362)
		Franz Ferdinand (318)

		99
		Mary J. Blige (5418)
		Aaliyah (4980)
		R. Kelly (4592)
		Jill Scott (3742)
		Boyz II Men (3369)
		Erykah Badu (3346)
		Musiq (2433)
		Keith Sweat (2406)
		Lauryn Hill (2377)
		112 (2217)

		100
		Pendulum (6583)
		Skrillex (4659)
		Mt Eden (3818)
		Pendulum (2460)
		Nero (2177)
		The Glitch Mob (2150)
		Chase & Status (1732)
		Avicii (1537)
		Flux Pavilion (1379)
		Example (1005)

		101
		Andrés Calamaro (5981)
		Jorge Drexler (3972)
		Fito Páez (3305)
		Luis Alberto Spinetta (3080)
		Feist (2839)
		Charly García (2818)
		Kevin Johansen (2573)
		Soda Stereo (2107)
		Pedro Aznar (1520)
		Babasónicos (1481)

		102
		Neil Young (7275)
		The Kinks (5083)
		The Velvet Underground (3862)
		Crosby, Stills, Nash & Young (3754)
		Bob Dylan (2383)
		Joni Mitchell (1778)
		Jefferson Airplane (1306)
		Cream (1193)
		T. Rex (1164)
		The Rolling Stones (1107)

		103
		The Rolling Stones (18942)
		Dave Matthews Band (10853)
		Phish (7584)
		Grateful Dead (4866)
		Keiko Matsui (1837)
		Dave Matthews (1076)
		Keller Williams (490)
		Blur (452)
		Widespread Panic (434)
		moe. (395)

		104
		Alison Krauss & Union Station (1624)
		Tangerine Dream (1544)
		Howlin' Wolf (1486)
		Alison Krauss (1235)
		Aventura (973)
		Frank Reyes (871)
		The Stanley Brothers (674)
		Lester Flatt & Earl Scruggs (674)
		Gillian Welch (658)
		Scientist (629)

		105
		Alpha Blondy (1538)
		Tina Dickow (1349)
		Tiken Jah Fakoly (1192)
		Robyn (1124)
		Aqua (658)
		The White Panda (636)
		Fallulah (585)
		Nephew (539)
		Murray Gold (525)
		Powderfinger (509)

		106
		Michael Bublé (14776)
		Stevie Ray Vaughan (2113)
		Ed Sheeran
		(2065)
		Jamie Cullum (1026)
		Il Divo (1011)
		Lauryn Hill (808)
		Joe Walsh (774)
		Josh Groban (735)
		Macy Gray (715)
		Mika (660)

		107
		Weezer (6479)
		Eels (1975)
		Broken Social Scene (1848)
		The Shins (1694)
		Death Cab for Cutie (1440)
		of Montreal (1309)
		Clap Your Hands Say Yeah (1101)
		Rilo Kiley (994)
		Stars (914)
		Pinback (808)

		108
		Lady Gaga (6735)
		Tiësto (5695)
		Jackson 5 (1831)
		David Cook (1707)
		Adam Lambert (1162)
		鷺巣詩郎 (818)
		Little Brother (754)
		Lee DeWyze (656)
		David Archuleta (426)
		Richard O'Brien (409)

		109
		The Strokes (5970)
		Arctic Monkeys (4882)
		Incubus (3601)
		Andrea Bocelli (2052)
		McFly (1590)
		Big Country (946)
		Brazilian Girls (655)
		Nach (629)
		Nando Reis (599)
		Busted (589)

		110
		Caetano Veloso (3033)
		Chico Buarque (1913)
		Gilberto Gil (1527)
		Raul Seixas (1273)
		Maria Bethânia (1267)
		Jorge e Mateus (1142)
		Los Hermanos (1049)
		Engenheiros do Hawaii (1023)
		Armandinho (869)
		CéU (845)

		111
		Bob Marley (32678)
		Bob Marley & The Wailers (16360)
		Toots & The Maytals (3241)
		Black Uhuru (2872)
		Steel Pulse (1214)
		Jimmy Cliff (1110)
		Damian Marley (1049)
		Lee "Scratch" Perry (1046)
		Barrington Levy (871)
		King Tubby (755)

		112
		Iron & Wine (4124)
		Stars (2143)
		Chris Isaak (2032)
		The National (1780)
		Bon Iver (1665)
		Robert Cray (913)
		Pulp (783)
		Scott Johnson (695)
		Cloud Cult (571)
		Chicago Public Radio (541)

		113
		Metallica (15496)
		50 Cent (8250)
		Jimmy Buffett (6268)
		Taylor Swift (4742)
		Easy Star All-Stars (1073)
		The String Quartet (962)
		Dmitri Shostakovich (490)
		Gustav Mahler (388)
		Béla Bartók (337)
		Parkway Drive (320)

		114
		Dane Cook (2529)
		Kate Nash (2391)
		George Carlin (1865)
		Fergie (1620)
		Stravinsky (1429)
		Mitch Hedberg (1231)
		Richard Pryor (1069)
		Jim Gaffigan (998)
		Lewis Black (964)
		Bill Cosby (807)

		115
		Radiohead (8179)
		Black Sabbath (7728)
		Death Cab for Cutie (6695)
		Dream Theater (4053)
		Porcupine Tree (2481)
		Yngwie J. Malmsteen (1055)
		Pinetop Perkins (845)
		Liquid Tension Experiment (805)
		Europe (728)
		Ozzy Osbourne (498)

		116
		Bright Eyes (2478)
		Jadakiss (2001)
		M.I.A. (1608)
		Styles P (1458)
		MIA. (975)
		The LOX (750)
		LL Cool J (657)
		Die Antwoord (635)
		Beanie Sigel (589)
		Los Hermanos Rosario (492)

		117
		O.A.R. (3752)
		Dispatch (3186)
		CAKE (2873)
		Talking Heads (2743)
		Billy Idol (2341)
		Guster (1683)
		Michael Franti & Spearhead (1408)
		The John Butler Trio (1136)
		Rusted Root (888)
		Donavon Frankenreiter (679)

		118
		Jay Chou 周杰伦 (4713)
		Claude Debussy (2846)
		R.E.M. (2506)
		王力宏 (2081)
		陶喆 (1890)
		Astral Projection (1363)
		Hallucinogen (1050)
		張學友 (1015)
		周杰倫 (Jay Chou) (885)
		陳奕迅 (885)

		119
		Pearl Jam (13530)
		Soundgarden (6946)
		Matchbox Twenty (5671)
		Counting Crows (4898)
		Third Eye Blind (4328)
		Stone Temple Pilots (4294)
		The Goo Goo Dolls (3693)
		Audioslave (3270)
		Alice in Chains (3042)
		Eddie Vedder (2761)

		120
		Björk (5626)
		Tori Amos (2390)
		Jimmy Cliff (2017)
		PJ Harvey (1950)
		Nouvelle Vague (1587)
		Nick Drake (1345)
		Sarah McLachlan (1125)
		Fiona Apple (1107)
		CocoRosie (971)
		Jeff Buckley (962)

		121
		Infected Mushroom (3925)
		Hatsune Miku (1862)
		Faithless (1065)
		Linken Park (572)
		The Brand New Heavies (551)
		The KLF (505)
		Megurine Luka (449)
		Kagamine Len (438)
		Wolfsheim (430)
		vocaloid (Miku Hatsune) (428)

		122
		Tool (8441)
		Pantera (7692)
		Metallica (4456)
		Megadeth (4277)
		Slayer (3585)
		Iron Maiden (2994)
		Rage Against the Machine (2843)
		Sepultura (2719)
		Motörhead (2349)
		Alice in Chains (1820)

		123
		Eminem (62896)
		D12 (1373)
		50 Cent (1315)
		Dr. Dre (1166)
		The Rosenberg Trio (352)
		Micheal Jackson (296)
		The Section Quartet (283)
		Classical Masterpieces of the Millenium (220)
		Obie Trice (189)
		Lil Wayne (185)

		124
		Naruto (8966)
		These Animal Men (5934)
		梶浦由記 (4116)
		Gravitation (1677)
		See-Saw (1210)
		Ion Storm (1151)
		Bruce Faulconer (1086)
		???Y ?R?L (1018)
		Fruits Basket (1018)
		Gundam Wing (888)

		125
		Jason Mraz (12541)
		Def Leppard (2750)
		KISS (1020)
		Buckethead (521)
		Rosario Flores (506)
		Portugal. The Man (431)
		Isis (391)
		Pig Destroyer (221)
		Macabre (201)
		Hernan Cattaneo (173)

		126
		Sublime (22187)
		Slightly Stoopid (3504)
		Pepper (2560)
		Rebelution (2097)
		عمرو دياب (1930)
		The Expendables (1514)
		311 (1322)
		Amon Amarth (1013)
		The Dirty Heads (1004)
		Bushido (786)

		127
		Frédéric Chopin (21193)
		Wolfgang Amadeus Mozart (15969)
		Johann Sebastian Bach (14777)
		Ludwig van Beethoven (10698)
		Antonio Vivaldi (8925)
		Mozart (7017)
		Beethoven (6534)
		Bach (6322)
		Chopin (5830)
		Tchaikovsky (3315)

		128
		Kanye West (15151)
		Nirvana (12701)
		Michael Jackson (7723)
		Mariah Carey (4179)
		Micheal Jackson (1129)
		Beyond (576)
		Flo Rida (401)
		Kansas (347)
		Mase (342)
		Bob Sinclair (316)

		129
		Teoman (3127)
		Şebnem Ferah (2631)
		Duman (1380)
		Tarkan (671)
		MFÖ (661)
		Bülent Ortaçgil (653)
		Ezginin Günlüğü (599)
		Sezen Aksu (572)
		Yeni Türkü (552)
		Mor ve Ötesi (530)

		130
		Elvis Presley (4308)
		Various Artists (2277)
		Chuck Berry (2146)
		Buddy Holly (2020)
		Jerry Lee Lewis (1692)
		The Four Seasons (1654)
		Richard Clayderman (1479)
		The Beach Boys (1398)
		The Platters (1376)
		The Everly Brothers (1328)

		131
		Fleetwood Mac (5953)
		Carpenters (2876)
		Christmas (2830)
		Young Jeezy (2127)
		Trans-Siberian Orchestra (2086)
		Bing Crosby (1418)
		Celtic Thunder (1216)
		Vangelis (1162)
		101 Strings Orchestra (1116)
		Stevie Nicks (1111)

		132
		John Coltrane (6856)
		Sigur Rós (4748)
		Miles Davis (3590)
		Charlie Parker (3325)
		Chick Corea (1393)
		Dizzy Gillespie (1279)
		Duke Ellington (1100)
		Bill Evans (1094)
		Thelonious Monk (1061)
		Count Basie (999)

		133
		Frank Sinatra (10656)
		Disney (5647)
		Dean Martin (1709)
		Cam Ly (995)
		Bing Crosby & Frank Sinatra (986)
		Putumayo (900)
		Varios (874)
		Alan Menken (739)
		Anastasia (730)
		Sammy Davis Jr. (685)

		134
		Eric Clapton (7028)
		Joaquín Sabina (6856)
		The Who (6293)
		Rush (3860)
		Yes (1598)
		Sabina (1115)
		Joan Manuel Serrat (680)
		Genesis (498)
		Buddy Guy (464)
		Jethro Tull (407)

		135
		Kirk Franklin (5274)
		Whitney Houston (2679)
		Fred Hammond (2304)
		Serge Gainsbourg (1743)
		Yolanda Adams (1631)
		Randy Travis (1623)
		Jacques Brel (1015)
		Donnie McClurkin (847)
		Paris Combo (820)
		Israel & New Breed (773)

		136
		John Williams (14176)
		Hans Zimmer (13685)
		Howard Shore (8831)
		Ion Storm (7396)
		James Horner (3160)
		Ennio Morricone (2732)
		Danny Elfman (2387)
		James Newton Howard (2052)
		John Powell (1881)
		Hans Zimmer & James Newton Howard (1801)

		137
		Breaking Benjamin (8006)
		3 Doors Down (7462)
		Three Days Grace (6716)
		Shinedown (4848)
		Papa Roach (4085)
		Staind (3845)
		Skillet (3640)
		Linkin Park (3508)
		Hinder (3109)
		Theory of a Deadman (2662)

		138
		Evanescence (6424)
		Boyce Avenue (3964)
		Flyleaf (3559)
		Enigma (3338)
		Depeche Mode (2401)
		Within Temptation (1224)
		Genesis (933)
		Inti-Illimani (743)
		Dub FX (700)
		Citizen Cope (665)

		139
		A Day to Remember (8095)
		Bring Me the Horizon (3946)
		Insane Clown Posse (3819)
		Escape the Fate (3376)
		The Devil Wears Prada (2646)
		Twiztid (2136)
		Alesana (1873)
		brokeNCYDE (1691)
		Parkway Drive (1532)
		Chiodos (1482)

		140
		Jesus Culture (3269)
		Hillsong United (1893)
		Misty Edwards (1518)
		Phil Wickham (1184)
		Patrice (681)
		David Holmes (642)
		Toshihiko Sahashi (641)
		Kim Walker (605)
		Bethel Live (550)
		Hillsong (504)

		141
		Foo Fighters (5498)
		Bob Dylan (2854)
		The Ventures (2163)
		Matisyahu (1960)
		Jim Sturgess (1650)
		Across The Universe (1450)
		Gotye (1379)
		James Taylor (1105)
		Minus the Bear (1045)
		Umphrey's McGee (956)

		142
		mc chris (1901)
		Stephen Lynch (1868)
		Ween (1832)
		They Might Be Giants (1549)
		Jonathan Coulton (1546)
		Rucka Rucka Ali (1375)
		Eclipse Music Group (1106)
		MC Frontalot (1081)
		"Weird Al" Yankovic (806)
		Bo Burnham (747)

		143
		Massive Attack (6745)
		Morcheeba (4295)
		Bonobo (3168)
		Portishead (3085)
		Zero 7 (2992)
		RJD2 (2989)
		DJ Shadow (2612)
		Audiomachine (1844)
		Orbital (1370)
		Lamb (1224)

		144
		Ramones (4780)
		Rancid (3722)
		NOFX (3485)
		The Misfits (3286)
		Dead Kennedys (1792)
		Clash (1522)
		Pennywise (1270)
		Bad Religion (1130)
		Black Flag (1104)
		Social Distortion (1080)

		145
		Dropkick Murphys (12249)
		The Pogues (7206)
		Flogging Molly (6682)
		The Dubliners (3794)
		The Irish Rovers (2150)
		Miranda (1533)
		Great Big Sea (1519)
		Gaelic Storm (1144)
		Buena Fe (1072)
		The Tossers (890)

		146
		Various (3086)
		Unknown Artist (2631)
		Alicia Keys (2492)
		VNV Nation (1790)
		Combichrist (1259)
		Kraftwerk (1237)
		Front Line Assembly (1083)
		Robin Thicke (926)
		Suicide Commando (793)
		Velvet Acid Christ (614)

		147
		Patricio Rey y sus Redonditos de Ricota (7441)
		Jimi Hendrix (6887)
		Los Piojos (4782)
		Divididos (3531)
		Bersuit Vergarabat (3151)
		Las Pastillas del Abuelo (3031)
		La Vela Puerca (2884)
		No te va Gustar (2720)
		La Renga (2458)
		Las Pelotas (1980)

		148
		Buddha Bar (CD Series) (2109)
		Deuter (876)
		Buddha Bar (784)
		Krishna Das (772)
		Deva Premal (761)
		Carter Burwell (746)
		R. Carlos Nakai (739)
		Karunesh (669)
		Ryan Adams (666)
		Robert Pattinson (609)

		149
		Linkin Park (16293)
		blink-182 (15155)
		Green Day (14383)
		Fall Out Boy (5152)
		Sum 41 (4585)
		"Weird Al" Yankovic (3673)
		Simple Plan (1462)
		Blink 182 (1312)
		The Offspring (1012)
		Thin Lizzy (945) 

