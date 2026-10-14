# Error analysis, run 2026-10-14

Status: **Verified (run 2026-10-14)**. Spam is the positive class; a *false negative* is spam that was let through, a *false positive* is ham that was flagged. Test messages: 1,672, from the notebook's split.

The messages below are quoted from `data/spam.csv` (UCI SMS Spam Collection, CC BY 4.0), which is already public in this repository. They are shown exactly as they appear in the dataset, in full: nothing is masked or shortened, so phone numbers, shortcodes and names are those of the public corpus (line breaks become spaces). Scores are the model's spam score (probability, or SVM margin where the heading says so).

## Baseline against the best model (Verified (run 2026-10-14))

Of the 60 spam messages the baseline missed, the best model also misses 11 and catches 49. The best model misses 0 spam messages that the baseline caught.

## Baseline from the notebook: Multinomial Naive Bayes, word TF-IDF

Counts: 60 false negatives, 0 false positives. (Verified (run 2026-10-14))

### What the mistakes look like (Verified (run 2026-10-14))

| Group | Messages | Median length (chars) | With a link | With a phone-style number | With a currency sign | Status |
|---|---|---|---|---|---|---|
| missed spam (false negatives) | 60 | 134 | 13% | 32% | 20% | Verified (run 2026-10-14) |
| caught spam (true positives) | 178 | 149 | 16% | 62% | 38% | Verified (run 2026-10-14) |
| flagged ham (false positives) | 0 | n/a | n/a | n/a | n/a | Verified (run 2026-10-14) |
| correct ham (true negatives) | 1434 | 53 | 0% | 0% | 0% | Verified (run 2026-10-14) |

### False positives: ham flagged as spam (0, Verified (run 2026-10-14))

None.

### False negatives: spam let through (60, Verified (run 2026-10-14))

- score 0.002: `Sorry I missed your call let's talk when you have the time. I'm on 07090201529`
- score 0.020: `Babe: U want me dont u baby! Im nasty and have a thing 4 filthyguys. Fancy a rude time with a sexy bitch. How about we go slo n hard! Txt XXX SLO(4msgs)`
- score 0.071: `CALL 09090900040 & LISTEN TO EXTREME DIRTY LIVE CHAT GOING ON IN THE OFFICE RIGHT NOW TOTAL PRIVACY NO ONE KNOWS YOUR [sic] LISTENING 60P MIN 24/7MP 0870753331018+`
- score 0.077: `More people are dogging in your area now. Call 09090204448 and join like minded guys. Why not arrange 1 yourself. There's 1 this evening. Aå£1.50 minAPN LS278BB`
- score 0.090: `accordingly. I repeat, just text the word ok on your mobile phone and send`
- score 0.091: `Back 2 work 2morro half term over! Can U C me 2nite 4 some sexy passion B4 I have 2 go back? Chat NOW 09099726481 Luv DENA Calls å£1/minMobsmoreLKPOBOX177HP51FL`
- score 0.093: `Block Breaker now comes in deluxe format with new features and great graphics from T-Mobile. Buy for just å£5 by replying GET BBDELUXE and take the challenge`
- score 0.095: `thesmszone.com lets you send free anonymous and masked messages..im sending this message from there..do you see the potential for abuse???`
- score 0.103: `Hi, the SEXYCHAT girls are waiting for you to text them. Text now for a great night chatting. send STOP to stop this service`
- score 0.114: `You have 1 new message. Please call 08718738034.`
- score 0.114: `You have 1 new message. Please call 08712400200.`
- score 0.119: `TBS/PERSOLVO. been chasing us since Sept forå£38 definitely not paying now thanks to your information. We will ignore them. Kath. Manchester.`
- score 0.130: `CLAIRE here am havin borin time & am now alone U wanna cum over 2nite? Chat now 09099725823 hope 2 C U Luv CLAIRE xx Callså£1/minmoremobsEMSPOBox45PO139WA`
- score 0.166: `Hi if ur lookin 4 saucy daytime fun wiv busty married woman Am free all next week Chat now 2 sort time 09099726429 JANINExx Callså£1/minMobsmoreLKPOBOX177HP51FL`
- score 0.170: `500 free text msgs. Just text ok to 80488 and we'll credit your account`
- score 0.172: `LookAtMe!: Thanks for your purchase of a video clip from LookAtMe!, you've been charged 35p. Think you can do better? Why not send a video in a MMSto 32323.`
- score 0.173: `cmon babe, make me horny, *turn* me on! Txt me your fantasy now babe -) Im hot, sticky and need you now. All replies cost å£1.50. 2 cancel send STOP`
- score 0.185: `Hi babe its Chloe, how r u? I was smashed on saturday night, it was great! How was your weekend? U been missing me? SP visionsms.com Text stop to stop 150p/text`
- score 0.206: `How about getting in touch with folks waiting for company? Just txt back your NAME and AGE to opt in! Enjoy the community (150p/SMS)`
- score 0.207: `Burger King - Wanna play footy at a top stadium? Get 2 Burger King before 1st Sept and go Large or Super with Coca-Cola and walk out a winner`
- score 0.207: `Hi I'm sue. I am 20 years old and work as a lapdancer. I love sex. Text me live - I'm i my bedroom now. text SUE to 89555. By TextOperator G2 1DA 150ppmsg 18+`
- score 0.209: `Marvel Mobile Play the official Ultimate Spider-man game (å£4.50) on ur mobile right now. Text SPIDER to 83338 for the game & we ll send u a FREE 8Ball wallpaper`
- score 0.210: `SMS. ac JSco: Energy is high, but u may not know where 2channel it. 2day ur leadership skills r strong. Psychic? Reply ANS w/question. End? Reply END JSCO`
- score 0.221: `You have 1 new message. Call 0207-083-6089`
- score 0.229: `Latest News! Police station toilet stolen, cops have nothing to go on!`
- score 0.234: `Ur balance is now å£500. Ur next question is: Who sang 'Uptown Girl' in the 80's ? 2 answer txt ur ANSWER to 83600. Good luck!`
- score 0.240: `Hungry gay guys feeling hungry and up 4 it, now. Call 08718730555 just 10p/min. To stop texts call 08712460324 (10p/min)`
- score 0.243: `Goal! Arsenal 4 (Henry, 7 v Liverpool 2 Henry scores with a simple shot from 6 yards from a pass by Bergkamp to give Arsenal a 2 goal margin after 78 mins.`
- score 0.260: `Please CALL 08712402972 immediately as there is an urgent message waiting for you`
- score 0.269: `Bloomberg -Message center +447797706009 Why wait? Apply for your future http://careers. bloomberg.com`
- score 0.269: `Bloomberg -Message center +447797706009 Why wait? Apply for your future http://careers. bloomberg.com`
- score 0.275: `ringtoneking 84484`
- score 0.290: `You have 1 new voicemail. Please call 08719181513.`
- score 0.290: `You have 1 new voicemail. Please call 08719181513.`
- score 0.294: `Your free ringtone is waiting to be collected. Simply text the password \MIX\" to 85069 to verify. Get Usher and Britney. FML`
- score 0.294: `Your free ringtone is waiting to be collected. Simply text the password \MIX\" to 85069 to verify. Get Usher and Britney. FML`
- score 0.296: `You will be receiving this week's Triple Echo ringtone shortly. Enjoy it!`
- score 0.298: `A link to your picture has been sent. You can also use http://alto18.co.uk/wave/wave.asp?o=44345`
- score 0.298: `Your next amazing xxx PICSFREE1 video will be sent to you enjoy! If one vid is not enough for 2day text back the keyword PICSFREE1 to get the next video.`
- score 0.302: `You can donate å£2.50 to UNICEF's Asian Tsunami disaster support fund by texting DONATE to 864233. å£2.50 will be added to your next bill`
- score 0.319: `Want explicit SEX in 30 secs? Ring 02073162414 now! Costs 20p/min`
- score 0.353: `Sexy Singles are waiting for you! Text your AGE followed by your GENDER as wither M or F E.G.23F. For gay men text your AGE followed by a G. e.g.23G.`
- score 0.354: `Sorry! U can not unsubscribe yet. THE MOB offer package has a min term of 54 weeks> pls resubmit request after expiry. Reply THEMOB HELP 4 more info`
- score 0.364: `You won't believe it but it's true. It's Incredible Txts! Reply G now to learn truly amazing things that will blow your mind. From O2FWD only 18p/txt`
- score 0.365: `Hey I am really horny want to chat or see me naked text hot to 69698 text charged at 150pm to unsubscribe text stop 69698`
- score 0.370: `Thank you, winner notified by sms. Good Luck! No future marketing reply STOP to 84122 customer services 08450542832`
- score 0.400: `Dorothy@kiefer.com (Bank of Granite issues Strong-Buy) EXPLOSIVE PICK FOR OUR MEMBERS *****UP OVER 300% *********** Nasdaq Symbol CDGT That is a $5.00 per..`
- score 0.402: `Dear Voucher holder Have your next meal on us. Use the following link on your pc 2 enjoy a 2 4 1 dining experiencehttp://www.vouch4me.com/etlp/dining.asp`
- score 0.403: `Bored of speed dating? Try SPEEDCHAT, txt SPEEDCHAT to 80155, if you don't like em txt SWAP and get a new chatter! Chat80155 POBox36504W45WQ 150p/msg rcd 16`
- score 0.417: `You'll not rcv any more msgs from the chat svc. For FREE Hardcore services text GO to: 69988 If u get nothing u must Age Verify with yr network & try again`
- score 0.431: `Customer service annoncement. You have a New Years delivery waiting for you. Please call 07046744435 now to arrange delivery`
- score 0.440: `We currently have a message awaiting your collection. To collect your message just call 08718723815.`
- score 0.456: `3. You have received your mobile content. Enjoy`
- score 0.471: `SMS. ac Blind Date 4U!: Rodds1 is 21/m from Aberdeen, United Kingdom. Check Him out http://img. sms. ac/W/icmb3cktz8r7!-4 no Blind Dates send HIDE`
- score 0.478: `RECPT 1/3. You have ordered a Ringtone. Your order is being processed...`
- score 0.479: `Dear Subscriber ur draw 4 å£100 gift voucher will b entered on receipt of a correct ans. When was Elvis Presleys Birthday? TXT answer to 80062`
- score 0.497: `YES! The only place in town to meet exciting adult singles is now in the UK. Txt CHAT to 86688 now! 150p/Msg.`
- score 0.497: `YES! The only place in town to meet exciting adult singles is now in the UK. Txt CHAT to 86688 now! 150p/Msg.`
- score 0.499: `Send a logo 2 ur lover - 2 names joined by a heart. Txt LOVE NAME1 NAME2 MOBNO eg LOVE ADAM EVE 07123456789 to 87077 Yahoo! POBox36504W45WQ TxtNO 4 no ads 150p.`
- score 0.499: `Hottest pics straight to your phone!! See me getting Wet and Wanting, just for you xx Text PICS to 89555 now! txt costs 150p textoperator g696ga 18 XxX`

## Best model by the selection rule: Linear SVM, word + char TF-IDF

Counts: 11 false negatives, 0 false positives. (Verified (run 2026-10-14))

### What the mistakes look like (Verified (run 2026-10-14))

| Group | Messages | Median length (chars) | With a link | With a phone-style number | With a currency sign | Status |
|---|---|---|---|---|---|---|
| missed spam (false negatives) | 11 | 137 | 9% | 9% | 18% | Verified (run 2026-10-14) |
| caught spam (true positives) | 227 | 149 | 16% | 56% | 34% | Verified (run 2026-10-14) |
| flagged ham (false positives) | 0 | n/a | n/a | n/a | n/a | Verified (run 2026-10-14) |
| correct ham (true negatives) | 1434 | 53 | 0% | 0% | 0% | Verified (run 2026-10-14) |

### False positives: ham flagged as spam (0, Verified (run 2026-10-14))

None.

### False negatives: spam let through (11, Verified (run 2026-10-14))

- score -0.783: `Sorry I missed your call let's talk when you have the time. I'm on 07090201529`
- score -0.681: `TBS/PERSOLVO. been chasing us since Sept forå£38 definitely not paying now thanks to your information. We will ignore them. Kath. Manchester.`
- score -0.394: `Latest News! Police station toilet stolen, cops have nothing to go on!`
- score -0.335: `ringtoneking 84484`
- score -0.320: `LookAtMe!: Thanks for your purchase of a video clip from LookAtMe!, you've been charged 35p. Think you can do better? Why not send a video in a MMSto 32323.`
- score -0.266: `thesmszone.com lets you send free anonymous and masked messages..im sending this message from there..do you see the potential for abuse???`
- score -0.261: `Burger King - Wanna play footy at a top stadium? Get 2 Burger King before 1st Sept and go Large or Super with Coca-Cola and walk out a winner`
- score -0.229: `accordingly. I repeat, just text the word ok on your mobile phone and send`
- score -0.064: `You can donate å£2.50 to UNICEF's Asian Tsunami disaster support fund by texting DONATE to 864233. å£2.50 will be added to your next bill`
- score -0.026: `Goal! Arsenal 4 (Henry, 7 v Liverpool 2 Henry scores with a simple shot from 6 yards from a pass by Bergkamp to give Arsenal a 2 goal margin after 78 mins.`
- score -0.022: `You will be receiving this week's Triple Echo ringtone shortly. Enjoy it!`
