# Runtime word blocklists

Each `{lang}.txt` file is a UTF-8 list with one blocked word per line. Blank
lines and lines beginning with `#` are ignored. The backend normalizes these
entries with the corresponding language module when it builds vocabulary
indexes at startup. Keep lists reviewed and maintain separate source/license
records if importing third-party lists.
