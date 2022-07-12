# UNIX and Linux

## Commands

You can use `man <command>` to view the document of a command. Also see the cheatsheet [here](https://www.guru99.com/linux-commands-cheat-sheet.html).

Besides, you can clean the screen by `Ctrl + L`.

### File system

| Command | Function                                   |
| ------- | ------------------------------------------ |
| ls      | list files and folders (see also `tree`)   |
| cd      | change current directory                   |
| pwd     | show current directory                     |
| mkdir   | create directory                           |
| rm      | delete files or folders (see also `rmdir`) |
| mv      | move or rename file                        |
| cp      | copy                                       |
| du      | size of files and folders                  |

### I/O

| Command | Function                                                     |
| ------- | ------------------------------------------------------------ |
| cat     | show file content / create new file / concatenate files      |
| more    | show file content page by page (also see `less`)             |
| head    | show the head of a file (also see `tail`)                    |
| echo    | output a message to STDOUT; can used together with `>>` or `>` to write into files |

Note:

- You can use `head -n <LineIndex> <filename> | tail -n 1` to get a specific line in a file.
- `>` overwrites existing content while `>>` appends to it.

### Search

| Command | Function                                                     |
| ------- | ------------------------------------------------------------ |
| locate  | Search according to a maintained database; fast but not necessarily exhausted |
| find    | Search and process                                           |
| grep    | Global Regular Expression Print (to STDOUT)                  |

### Process Management

| Command | Function                                           |
| ------- | -------------------------------------------------- |
| ps      | View processes' status (see also `jobs`)           |
| fg      | Bring suspended jobs to foreground (see also `bg`) |
| kill    | Kill specific process                              |

From FSL:

- To kill a job that is running (in the "foreground") in the terminal just type `Ctrl-c`
- To get a list of jobs running the background, type `jobs`
- To bring something from running in the background to running in the foreground, type `fg` if it is the last job that was started.
- To force a job that is already running (in the foreground) to be in the background, type `Ctrl-z` and then `bg`. Note that `Ctrl-z` on its own will give you the terminal prompt back but the job will be sleeping, not running in the background, until you do `bg`.

### Other Commands

- `tar`: archive management
  - Modes (only 1 of the following can be used!)
    - `-c`: packing
    - `-t`: list the content of a tarfile
    - `-x`: unpacking
  - `-C <path>`: output to certain path
  - `-f <filename>`: specify filename (no more parameter is allowed between `-f` and `<filename>`!)
  - `-z`: compress (or uncompresse) with gzip (for `.tar.gz` files)

- Operators to run two command in one line:

  - `;` or `&&` (logical AND) or `||` (logical OR)

  - If you just want to execute two commands at once, `&&` is recommended since the second command will only be executed if the first one successfully returns. For example, you may cause disaster if you want to remove the content of a directory but type it wrong:

    ```bash
    cd <Typo> ; rm -rf *
    ```

    which will delete everything in your system.
    
  - Also see https://www.tecmint.com/chaining-operators-in-linux-with-practical-examples/

- Line break: use back slash `\` to continue your command on a new line

- tmux: allowing the client to leave a session running on the remote after logout; multi-session split window

  - Install: `apt-get install tmux`
  - Create a session (openning a tmux window): `tmux new -s mySession`
  - Creating another window: `Ctrl + B` and then `c`
  - Switching windows: `Ctrl + B` and then `0-9`
  - Exiting tmux while leaving the session running: `Ctrl + B` and then `d`
  - Closing the session completely: `exit`
  - Viewing current tmux sessions: `tmux ls`
  - Reopening the running session: `tmux a -t mySession`


## Filesystem Hierarchy Standard (FHS)

### Folders in root Directory `/`

| Folder                             | Functions / Contents                                         |
| ---------------------------------- | ------------------------------------------------------------ |
| bin                                | System command binaries, e.g. `ls`                           |
| boot                               | Static files of boot loaders, e.g. kernal files              |
| dev                                | Device files                                                 |
| etc                                | Static configuration files                                   |
| home                               | User home directories, containing folders named after each user |
| lib (and optionally lib32 / lib64) | Shared libraries                                             |
| media                              | Mount (挂载) point for removable media (e.g. CD)             |
| mnt                                | Mount point for removable filesystems (e.g. U-disk)          |
| opt                                | Add-on application packages                                  |
| root                               | Home directory of the root user                              |
| run                                | Run-time system information                                  |
| sbin                               | System binaries                                              |
| srv                                | Services data                                                |
| tmp                                | Temporary files                                              |
| usr                                | Unix software resources (read-only, sharable between OS)     |
| var                                | Variable data files                                          |

Other directories under root:

| Folder         | Functions / Contents                                         |
| -------------- | ------------------------------------------------------------ |
| lost+found     | File fragmentations due to crash or unexpected shutdown, usually empty |
| proc (and sys) | Mapping of some system information in the memory (not hard drive!) |

Notes:

- When you log in as a user (e.g. `chen`), your home directory (here `/home/chen/`) can be shortened as `~`
- Softwares are usually installed in `/opt`  or `/usr/local`, exactly which can be tricky:
  - `/usr/local` contains folders like `/bin`, `/sbin`, thus suitable for softwares not managed by the system packager, but still following the standard unix deployment rules (e.g. the ones your build with `make`)
  - softwares installed in `/opt` will have their own directory and may not follow these rules
- See [here](https://refspecs.linuxfoundation.org/FHS_3.0/fhs/ch04.html) for more about the `/usr` directory

## Terminal, Console, TTY, Shell?

- History:
  - In the very early UNIX computers, the machine is huge and usually only the administrator will control it directly through some hardware (a console)
  - Since UNIX is the first multi-user computer structure, every user is allowed to interact with the machine (terminal) using a device to send command and receive output
  - This machine is often a teletypewriter (TTY)
  - In modern system, TTY, terminal and console are the same, which is an interface to send and receive texts
- Terminal emulator:
  - The word "terminal" in modern use actually means "terminal emulator", which is a program that simulates the behavior of traditional (physical) terminals and interacts with the shell
  - Terminal emulator usually uses command-line-interface (CLI), but the program itself usually runs in a graphical-user-interface (GUI) system (e.g. Windows)
  - Terminal emulator can provide functions like changing colors and font, command history, auto complete, copy and paste, etc.
  - Some examples:
    - Linux: Gnome-terminal, Konsole
    - Mac: iTerm2
    - Windows: Win32 Console, Windows Terminal
- Shell:
  - A shell is the outmost layer around the operating system, which exposes the system's services to the user
  - Shell interprets users' inputs, calls kernal APIs (and sometimes other applications) and handles the outputs. For example, when you input the command `cat foo.txt` in a shell, it will call `cat.exe` and the latter will call kernal APIs like `open`
  - Shell is just a special application. It calls system APIs just like other applications, and thus replacable by similar applications (e.g., Windows Explorer can be replaced by `cmd.exe`)
  - Shells can use terminals to accept user inputs and provide outputs, e.g. PowerShell use Win32 console as its interface
  - Examples:
    - UNIX: sh
    - Linux: bash
    - Windows: command prompt (`cmd.exe`), PowerShell

## Windows Subsystem for Linux (WSL)

- Essentially a virtual machine running Linux

- Can be installed by typing this command in Powershell (administrators):

  ```bash
  wsl --install -d <Distribution Name>
  ```

  Replace `<Distribution Name>` with the Linux distribution you want, e.g. `Ubuntu-18.04` (as used below). If you dismiss the `-d` parameter, it will be the latest version of Ubuntu.

  See [Docs from Microsoft](https://docs.microsoft.com/en-us/windows/wsl/install) for details.

- [Windows Terminal](https://docs.microsoft.com/en-us/windows/terminal/get-started) allows you handle multiple CLI programs (in WSL and Windows) at the same time.

- File systems:

  - In WSL, the windows file system will be regarded as mounted drives (like an U-disk), thus will be put in `/mnt/` directory (e.g. `/mnt/c/Windows/`)
  - In Windows, the WSL file system will be regarded as a network drive, and can be viewed by inputing the url in Windows file explorer: `\\wsl$\<Your Linux Version>\`
  - Therefore, it will be faster to store your projects in the file system that you will be working on, rather than accessing them through the method above

- Using GUI:

  - You can use `VcXsrv` on Windows to display WSL GUI applications.
  
  - Follow the instructions of FSL:
  
    - Install [VcXsrv](https://sourceforge.net/projects/vcxsrv/files/latest/download), which will install a launcher (to `vcxsrv.exe`) called `XLaunch`
  
    - Open your Ubuntu Shell and enter these commands:
  
      ```bash
      echo "export DISPLAY=\$(grep nameserver /etc/resolv.conf  | awk '{print \$2; exit}'):0" >> ~/.bashrc
      echo "export LIBGL_ALWAYS_INDIRECT=1" >> ~/.bashrc
      ```
  
      which add some settings into you `~/.bashrc` file so that VcXsrv can work.
  
    - To use GUI in WSL, open XLaunch and select the options you want (I usually use `multiple windows` for display), but make sure to:
  
      - **Disselect** `Native OpenGL`
      - **Select** `Disable access control`
  
      in the `Extra settings` panel.
  
    - Besides, you make need to trust VcXsrv in your firewall.
  
  - You can save the configurations and use command line option `-run <Config>` to skip the setting and start the server silently. More exactly:
  
    - Save it to where VcXsrv is installed (so you don't need to specify the path in `<Config>`), usually `%ProgramFiles%\VcXSrv\` (i.e., `C:\Program Files\VcXsrv`)
    - Find the shortcut for XLaunch (or create one), go to Properties-Target and add `-run <Config>` after `"C:\Program Files\VcXsrv\xlaunch.exe"`
  
  - Besides, if you change the zooming ratio in Windows, the GUI display may be blurred. In this case, you need to open the properties of the shortcut, and select "override high DPI" as "Application" in the compatibility mode.
  
- Bugs:

  - You may find the following error when you launch the terminal:

    ```bash
    mkdir: cannot create directory '/home/<User Name>/.cache': Permission denied
    -bash: /home/<User name>/.cache/wslu/integration.sh: Permission denied
    ```

    - If you check the contents of your home directory with `ls -la ~`, you will find that `.cache` (and maybe also `.conda`) is owned by root, rather than you
    - Therefore, the Ubuntu for WSL integration script cannot write into this directory
    - It seems that this problem is caused by `sudo apt update` and `sudo apt upgrade`
    - Solution: delete this directory (anyway it's just cache) by `sudo rm -r ~/.cache`
    
  - VPN:

    - using Cisco Annyconnect for Linux in WSL2 will cause problem in DNS resolve
    - solution:
      - use VPN on Windows instead
      - you must connect to the VPN **after** lauching WSL!

## Remote development

- Basics:
  - The high-performance computing (HPC) clusters are usually Linux servers equipped with some computing job managment softwares
    - WUSM CHPC uses Slurm on Centos 8 (Update: it's Rock 8.5 now)
    - WUSTL RIS uses LSF
  - End-users usually use SSH to login to the server remotely. They:
    - have their own `$HOME` directory (`/home/username`) as the main workspace, and sometimes another directory for temporary usage (`/scratch/username` for CHPC)
    - have limited access to system directories:
      - e.g., they can run the commands in `/usr/bin`
      - but they cannot use `sudo` or write into system directories
  - Users submit bash scripts and specify the dependencies to the management system to perform the computation
    - Due to the possible conflict between dependencies for different jobs, the server usually manages an environment for each job
    - One traditional way (as it is in CHPC) is to prepare all possible versions of a dependency in the server, and use `module` to specify the environment variables for each job
    - Another way (as it is in RIS) is to use `docker` to encapsule all dependencies of a job into a standalone, platform-independent *container*
- Some common errors:
  - Permission denied even after entering correct password in VS Code Remote SSH (+ Linux ssh can work but show message "no more processes"):
    - Reason: someone is running big jobs and reached the maximum number of allowed processes on the server
    - Solution: shout out and wait
- `module`:
  - Use `module avail <package>` to see the available versions of a package
  - Use `module spider <RegExp>` to search for all available packages
  - Use `module load <package/version>` to load a specific version of a package into your environment
  - Use `module unload <package/version>` or `module del <package/version>` to unload
- `ssh` and `scp`:
  - Usually you need to start the ssh service by `sudo service ssh start`
  - Login to remote server: `ssh [parameters] [user@]host[:[port]] [command]`
    - frequently-used parameters:
      - `-p` (lowercase): specify the port to connected to (can also use `[user@]host:[port]` instead), by default port 22
      - `-Y`: enable X11 forwarding. You can run GUI on the server through X11 with this option.
    - `[command]` is executed on the remote host rather than local terminal
  - Logout: `logout` or `exit`
  - File transmission: `scp [parameters]  [[user@]host1:]file1 ... [[user@]host2:]file2`
    - `scp` use SSH protocol for security
    - frequently-used parameters:
      - `-P` (uppercase): specify the port to connected to (by default also port 22)
      - `-r`: recursively copy the whole folder and its contents
    - Examples:
      - Upload a local file to a remote directory: `scp <local_file> <user>@<host>:<directory>`
      - Same as above but change the name of the uploaded file: change `<directory>` to `<directory>/<new_filename>`
      - Upload a local folder to a remote directory: `scp -r <local_folder> <user>@<host>:<directory>`
      - Download is similar, just switching the last two parameters
- Python:
  - Basically you just need to activate one version of Python by `module load python/x.x.x`, and then do the following jobs just like you have Anaconda installed
  - The only difference: after you create an environment, activate it by `source activate xxx`, not `conda activate xxx` (but deactivation and installation etc. are the same)
- R:
  - Setup:
    - Follow [CHPC's instruction](https://sites.wustl.edu/chpc/resources/software/r/) to set up an conda environment and install R packages.
    - Install [VS Code R](https://marketplace.visualstudio.com/items?itemName=Ikuyadeu.r) and [R debugger](https://github.com/ManuelHentschel/VSCode-R-Debugger) in VS Code
    - Find `r.rpath.linux` and `r.rterm.linux` in VS Code settings and change both to the path to R executable (`which R`, make sure to expand `~` to `/home/username`)
    - Open the R terminal in your activate conda environment and:
      - install [languageserver](https://github.com/REditorSupport/languageserver) by `install.packages("languageserver")`
      - install [vscDebugger](https://github.com/ManuelHentschel/vscDebugger) by `remotes::install_github("ManuelHentschel/vscDebugger")`. Note that the [recommended way](https://marketplace.visualstudio.com/items?itemName=RDebugger.r-debugger) is not feasible for CHPC.
    - You may also want to install [httpgd](https://github.com/nx10/httpgd) for faster and better plotting: `install.packages("httpgd")`
    - You may also want to install [radian](https://github.com/randy3k/radian) for autocomplete and other functions in the R terminal:
      - `pip3 install --user radian` to install `radian` in `~/.local/bin`
      - In VS Code, change the path `r.rterm.linux` to `/home/yourname/.local/bin/radian` (don't change `r.rpath.linux`!)
      - Delete `--no-save` and `--no-restore` in `r.rterm.option` (because `radian` has these parameters as default already)
      - Set `r.bracketedPaste` as `true` to allow bracket-paste mode
  - Interaction:
    - In order to plot and explore your data, you need to use an interactive terminal instead of VS Code debugger
    - Select the code to execute and then press Ctrl+Enter to send them to an R interactive terminal, then you will see the variables in the R workspace panel on the left
    - You can examine the dataFrames and plots by executing `View(myvar)` in the interactive terminal
  - Bugs:
    - When you use `parallel()` for parallelization, use vanilla R instead of Radian, otherwise it will break



## Bash

- You need to specify the shell to use for your script at the begining. For example, to use `bash`:

  ```bash
  #!/bin/bash
  echo "Hello world!"
  ```

- How to run:

  - Save the script with postfix `.sh` (it's only a convention, not a requirement) 
  - Use `chmod +x <filename>` to make your script executable
  - Run it either by it's full name, or run the shell with the script as a parameter.
  
- Variable:

  - Definition and assignment:
    - NO space between variable name and assignment operator `=`!
    
    - can be defined implicitly, like `for myChar in "sdfsdf" `
    
    - can be defined as the output of certain commands:
    
      ```bash
      myVar=$(commands)
      myVar=$(commands [command opt ...] param1 param2 ...)
      myVar=`commands`
      myVar=`commands [command opt ...] param1 param2 ...`
      ```
    
  - Reference:
    - when using a variable `var1`, you should put the variable name [in a curly bracket] after a dollar sign
    - e.g., `echo ${var1}`
    
  - Read only: you can set a variable to read-only by `readonly var_name` (no dollar and bracket, similar below)

  - Delete: `unset var_name`

- String:

  - When your string doesn't contain spaces (which is often the case), you actual don't need to quote it

  - When the string contains spaces, normally we use `""`, which can contain variable reference

  - Content of a string variable cannot be modified after definition.

  - Indexing:
    - starting from 0
    - `${var1:i}` substring starting from index i to the end
    - `${var1:i:j}` substring of length j starting from index i (note: it's NOT i to j or j-1!)
    - `${var1::j}` substring of length j starting from index 0

  - `${#var1}` returns the length (can not use together with indexing)

  - Concatenation:
    - just put variables and strings together without spaces
    - e.g. `"Hello "${var1}" world""!"` is the same as `"Hello ${var1} world!"`

  - Paths:

    - You can use `*`, `?` or so to find out the filenames directly without any extra command:

      ```bash
      allMkFile=(./*.mk)  # Get a list of all .mk files in current directory
      ```

    - You can check whether a directory exists by the expression `[ -d "${myPath}" ]` (must keep the spaces! Otherwise `[-d` will be recognized as one token)

- Array:

  - only 1D array; NO restriction on array size
  - can use ANY non-negative integer as indices; indices can be NOT consecutive
  - definition:
    - `myArr=(var1 var2 var3 var4)`, or
    - `myArr[0]=var1; myArr[100]=var2; myArr[1]=var3`
  - indexing:
    - `${myArr[i]}` the element at index i (if there's no such element, no error and returns nothing)
    - `${myArr[@]}` all elements, in ascending indices
    - `${myArr[@]:i:j}`
      - (at most) j elements starting from index i
      - note that it is NOT the elements with index in range (i, i + j - 1), since index can be inconsecutive!
      - will return less that j elements (or even nothing) if there's no j elements with index larger or equal to i
    - `${myArr[@]:i}` and `${myArr[@]::j}`: similar to that in string (but note the difference due to inconsecutive indices)
  - `${#myArr[@]}` is the length (number of elements) of the array

- Expression:

  - Bash doesn't support many arithmatic and logical operations. You need to call utilities like `expr`, `test` and `[[ ... ]]` to calculate the expressions.

  - Arithmatic operation:
  
    ```bash
    a=1
    b=2
    c=`expr $a + $b`
    d=`expr $a \* $b`
    e=`expr $b % $a`   # mod
    ```

    Note:

    - MUST have spaces between operators and variables/numbers!
    - MUST have escape symbol before `*` (but not others)!
  
  - Boolean operation in bash:
  
    - There is NO bool type or keyword in bash!
  
    - `if, &&, ||, !` takes COMMANDs as its arguments and generates the result according to whether these commands succeed or fail
  
      - `true` and `false` (with or without quotes - after all, commands are just strings) are two commands that always succeeds or always fails
  
      - a command succeeds when it returns (exits with) **0**, and fails otherwise
  
      - `0` itself is not a command, so you cannot use it in logical expression
  
      - You can assign `true` and `false` to variables and use them in logical expression.
  
      - You can connect commands using `&&, ||` based on their short-circuiting feature:
  
        ```bash
        COMMAND && echo "Succeeded!" || echo "Failed!"
        ```
  
    - `test, [], [[]]` takes operators and operands as arguments
  
    - Examples:
  
      ```bash
      myVar="false"
      myFunc() { return 0; }
      
      if false ; then echo "True"; else echo "False"  # False
      if ${myVar} ; then echo "True"; else echo "False"  # False
      if [ false ] ; then echo "True"; else echo "False"  # True, since "false" is a non-empty string
      if myFunc ; then echo "True"; else echo "False"  # True
      if 0 ; then echo "True"; else echo "False"  # Error "0: command not found" and then "False"
      ```
  
  - Logical expression with `test expression` or `[ expression ]`
  
    - Fun fact:
      - `]` is a normal character, but `[` is not. It is a command!
      - `/usr/bin/[` is exactly the same as `/usr/bin/test`
      - when the file is called with name `[` it will search for paired `]`, but not if called with name `test`
      
    - Note:
      - Don't forget the spaces between expression and `[]`
      
      - ALWAYS quote your string variables otherwise empty string my give you an error:
      
        ```bash
        myVar=""
        [ ${myVar} = "" ] && echo "True"  # -bash: [: =: unary operator expected
        [ "${myVar}" = "" ] && echo "True"  # True
        [[ ${myVar} = "" ]] && echo "True"  # True
        ```
      
      - Also, make sure don't save your files carelessly with name `test`.
      
    - Number comparison: using `-eq`, `-gt`, `-le` instead of `==`, `>`, `<=`, etc.
  
    - Logical operator: using `-a, -o, !` instead of `&&, ||, !`
  
    - String test:
      - `=` or `==` whether two strings are the same (similarly `!=`)
      - `-z` whether a string has length 0
      - `-n "${str}"` or `${str}` whether a string has non-zero length
      - `-n ${str}` will always return true!
      
    - File test:
      - `-d xxx` whether `"xxx"` is a directory
      - `-f xxx` whether it is a file
      - ...
  
  - Logical operation with `[[ ... ]]`:
  
    - Only available in bash but not other shells
    - Similar to `[]`, but can use `&&, ||`
    - Can also use `<, >`, but it's ASCII comparison instead of numerical!
    - Can handle empty string variables without quotation
  
- Flow control:

  - If block:

    ```bash
    if condition1
    then
        command1
    elif condition2 
    then 
        command2
    else
        commandN
    fi
    ```

    Note: Bash doesn't allow empty clause. So if you don't need anything for `elif` and `else` just don't write them.

    In one line:

    ```bash
    if condition; then command1; command2; fi
    ```

  - While loop:

    ```bash
    while condition
    do
        command
    done
    ```

  - For loop:

    ```bash
    for var in item1 item2 ... itemN
    do
        command1
        command2
        ...
        commandN
    done
    ```

    Or in one line:

    ```bash
    for var in item1 item2 ... itemN; do command1; command2… done;
    ```

    For example, to print out the filenames of all file under current directory with postfix `.mk`:

    ```bash
    for i in ./*.mk
    do
        echo $i
    done
    ```

  - Case command:

    ```bash
    case $variable in
    pattern-1)
      commands;;
    pattern-2 | pattern-3)
      commands;;
    pattern-N)
      commands;;
    *)
      commands;;
    esac
    ```

    Note: All line breaks are optional. `*)` (default case) is optional.

  - Break loops: `continue`, `break` and `exit <status>` just like other languages.

- Command line parameters:

  - Note: you may want to include these parameters in double quotes when using, in case that they contains spaces

  - Positional parameters:

    - `${0}` is the name (with path) of the current file
    - `${1}, ${2}, ${3}` and so are the positional parameters
    - `$#` is the number of parameters received
    - `$*` concates all parameters into one string
    - `$@` returns all parameters with quotes

  - Command line options (short):

    - `-a, -h, -z`, etc.

    - using build-in command `getopts`

    - usage:

      ```bash
      #!/bin/bash
      echo "OPTIND starts at $OPTIND"
      while getopts ":pq:" optname
        do
          case "$optname" in
            "p")
              echo "Option $optname is specified"
              ;;
            "q")
              echo "Option $optname has value $OPTARG"
              ;;
            "?")
              echo "Unknown option $OPTARG"
              ;;
            ":")
              echo "No argument value for option $OPTARG"
              ;;
            *)
            # Should not occur
              echo "Unknown error while processing options"
              ;;
          esac
          echo "OPTIND is now $OPTIND"
        done
      ```

    - explained:

      - `$OPTIND` is the index of the parameter to be processed next
        - starting from 1
        - increased by 1 after running `getopts` each time
        - `getopts` will return `false` when `$OPTIND > $#`
      - `getopts <optstring> opt_var` processed the next input parameter according to pattern `optstring`, storing the option name in `$opt_var` and corresponding argument (if any) in `$OPTARG`
      - `<optstring>`:
        - the colon at the begining indicates silent mode (don't throw error, here for illustration)
        - the letters are options (`-p` and `-q` here)
        - a colon following a letter indicates it that has argument
        - note: the options can be combined with any order, except that arguments must follow their corresponding option immediately (e.g. you can input `-pq blabla` but not `-qp`)
      - `$opt_var`:
        - will be the option (e.g. `"p"`) for a correct option
        - will be `":"` if the option is correct but no required argument is provided
        - will be `"?"` for unknow option (e.g. if the input option is `-a`). In this case the option name will be stored in `OPTARG` (e.g. `"a"`)

  - Command line options (long):

    - `--help`, etc.
    - using external tool `getopt`
