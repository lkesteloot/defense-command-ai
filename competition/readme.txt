This is verion 1 of the trs8bit 2024 competition toolkit.

Please see http://48k.ca/comp2024.html for the latest version and other info.

This file and the included examples should give you all you need to work
on a competition entry.  It'll show you how to write a BASIC program to
remotely control the included Defense Command game.

If you're on Windows and just want to try this out then:

	1. Get the latest trs80gp from http://48k.ca/trs80gp.html
	2. Put the windows/trs80gp.exe executable in this directory
	3. Type "run dots.txt" to see a second TRS-80 play randomly.
	4. Type "run ascii.txt" to see the enemy types show by the remote.
	5. Type "run play.txt" to control the game from a second TRS-80.
	6. Type "runc codots.txt" to see a CoCo show the game as colorful dots.

On other platforms you can probably easily convert the run.bat and runc.bat
files into shell scripts.  If not or you need other assistance please don't
hesitate to get in touch with george@48k.ca

You can also set things up manually.  I really don't recommend it as a way
to do development but if will show the theory of operation of the batch files.

	1. Run trs80gp in model 3 mode with no diskette: trs80gp -m3 -dx
	2. Use Keyboard -> Connection and:
		Enter 3000 in the port field.
		Select TCP/IP Listen in the dropdown
	3. Use Printer -> Connection and:
		Enter 4000 in the port field.
		Select TCP/IP Listen in the dropdown
	4. Press enter twice and File -> Run remdc.cas
	5. Run a second trs80gp in model 3 mode: trs80gp -m3 -dx
	6. Use Keyboard -> Connection and:
		Enter 4000 in the port field.
		Select TCP/IP Connect in the dropdown
		You should see something like "Connected to 127.0.0.1:4000
	7. Use Printer -> Connection and:
		Enter 3000 in the port field.
		Select TCP/IP Connect in the dropdown
		You should see something like "Connected to 127.0.0.1:3000
	4. Press enter twice and File -> Run dots.txt

After all that you should see Defense Command running on the first TRS-80
and dots.txt mirroring various game objects as single pixels on screen.

What's happening here is that remdc.cas is sending game object positions
and status to BASIC's keyboard input routine.  dots.txt is reading that
information and displaying it on screen.  After every frame it uses
LPRINT to send player ship control information to remdc.cas to move
the shift left and right, fire shots and activate ABMs (smart bombs).

The BASIC programs use INPUT to get the information and LPRINT to say
what should be done.  Machine language programs can do the same thing
by calling $49 to read each character (on Model 1 and 3) and sending
control output to the printer port.  Here's a little routine for the
Model 3 do send a character in C to the printer.

pout:	in		a,($f8)
		add		a,a
		jr		c,pout
		ld		a,c
		out		($f8),a
		ret

On the Colour Computer INPUT is used but PRINT #-2,"..." is the way to
do printer output.  Machine language programs can JSR [$A000] to read
input and write -2 to $6F and JSR [$A002] to send control information.

The BASIC programs are pretty straightforward except for some odd POKEs
in there.  Those are used to turn input echoing off.  Otherwise when the
game data is sent it will be echoed to the screen by BASIC.  This may be
handy but it will generally destroy whatever your program may be trying to
display.  If you hit BREAK on the "AI" side you'll likely see no output
messages because the suppression will be in effect.  You can paste in
the POKE to turn the echoing back on.  Or type it "blind" if you're really
proficient.

Also note that trs80gp is directing keyboard input into BASIC/the standard
DOS input routine.  It will allow normal emulator keyboard input to inject
characters.  Thus typing on the trs80gp running the "AI" may cause problems
with the data it is reading.  However, with some care it can work out and
if you PEEK at the keyboard like play.txt does you can manage interactive
input.

For the open part of the competition and program or environment that can
make TCP/IP connections can talk to the game running on trs80gp.

The Protocol
============

The game starts by waiting for the AI to send a random number seed in decimal
followed by a carriage return.  The game has random events but will proceed
in a deterministic manner when starting with the same seed and given the same
inputs.

Every frame the game outputs one or more lines of the form:

	N,T,X,Y

	N	object number
	T	type number of the object
	X	X axis position of the object
	Y	Y axis position of the object

The lines are sent in object number order from 0 up to 40.  Then there will be
a line with object number 128.  This represents the end of the frame and
signals the AI to send back what it wants the player to do.

The player control line will have 0 or more single character commands:

	<	move player ship to the left
	>	move player ship to the right
	F	fire a shot
	!	fire an ABM (smart bomb)

In other words, the game will accept these commands as the equivalent
key presses made by a player.  A shot won't fire if there is already
one in the air and other limitations imposed by the game.

This back and forth will continue until the game ends which is signaled
by a object record with number 129.  The AI may decide to quit or
re-initialize for a new game.  It will have to send a new random number seed
to start the next game.

There are 16 object types:

	0	dead (no object in this slot)
	1	player ship
	2	fuel can
	3	player shot
	4	enemy shot
	5	flagship
	6	slicer
	7	enemy 1
	..
	14	enemy 8
	15	solar waster

To make things faster no update is sent if an object has not changed since
the previous frame.  The X and Y positions appear to be the bottom left corner
of the object and may be outside the normal (0 .. 127, 0 .. 47) screen
positions.

The end of frame object record 128 contains player status information instead
of the usual object information:

	128,score,remaining ships,remaining ABMs

The various enemies re-use object slots are the die and respawn.  The objects
come in a particular order which may change in subsequent versions of the
wrapper but you may find it helpful to use them even at the risk of code
changes down the road.

	0			player's ship
	1			player's shot
	2 .. 11		fuel cans	
	12 .. 36	enemy ships
	37 .. 39	enemy shots

The game doesn't report on the enemy mothership.  It appears at the start of
the game for a limited time and never can be killed.  When it appears at the
end of the game there is nothing the player can do.  It sends out solar
wasters (type 15) which are also indestructible.

Bonus ships, bonus ABMs and ship loss will show up in record 128 changes.
A bonus ship might be missed if the player dies in the same frame but a
bonus ship is given at 10,000 points so it can still be detected that way.

There is no indication when a fuel cell has been caught.  Checking player
coordinates vs. can coordinates will be sufficient but the exact details
will need to be worked out by the AI programmer.
