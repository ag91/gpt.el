;;; gpt.el --- Run instruction-following language models -*- lexical-binding: t; -*-

;; Copyright (C) 2022 Andreas Stuhlmueller

;; Author: Andreas Stuhlmueller <andreas@ought.org>
;; Version: 1.1
;; Keywords: openai, anthropic, claude, language, copilot, convenience, tools
;; URL: https://github.com/stuhlmueller/gpt.el
;; License: MIT
;; SPDX-License-Identifier: MIT
;; Package-Requires: ((emacs "24.4"))

;;; Commentary:

;; This package defines a set of functions and variables for running
;; instruction-following language models like GPT-4 and Claude 3.5
;; Sonnet.  It allows the user to enter a command with history and
;; completion, and optionally use the current region as input.  The
;; output of the command is displayed in a temporary buffer with the
;; same major mode as the original buffer.  The output is streamed as
;; it is produced by the GPT process.  The user can enter a follow-up
;; command in the output buffer, which will provide the output, the
;; follow-up command to GPT as a new prompt.  The follow-up output
;; will be appended to the output buffer.  The user can view and
;; export the command history to a file.

(require 'savehist)

(savehist-mode 1)

(defvar gpt-command-history nil
  "A list of GPT commands that have been entered by the user.")

(defvar gpt-script-path (expand-file-name "gpt.py" (file-name-directory (or load-file-name buffer-file-name)))
  "The path to the Python script used by gpt.el.")

(defcustom gpt-model "gpt-4o"
  "The model to use."
  :type 'string
  :group 'gpt)

(defvar gpt-max-tokens "2000"
  "The max_tokens value used with the chosen model.")

(defvar gpt-temperature "0"
  "The temperature value used with the chosen model.")

(defcustom gpt-openai-key "NOT SET"
  "The OpenAI API key to use."
  :type 'string
  :group 'gpt)

(defcustom gpt-anthropic-key "NOT SET"
  "The Anthropic API key to use."
  :type 'string
  :group 'gpt)

(defcustom gpt-writerai-key "NOT SET"
  "The Writer API key to use."
  :type 'string
  :group 'gpt)

(defvar gpt-writerai-model-cache nil "Cache for writerai models.")

(defcustom gpt-api-type 'openai
  "The type of API to use."
  :options '(openai anthropic writerai)
  :type 'symbol
  :group 'gpt)

(defcustom gpt-python-path "python"
  "The path to your python executable."
  :type 'string
  :group 'gpt)

(defcustom gpt-use-named-buffers t
  "If non-nil, use named buffers for GPT output. Otherwise, use temporary buffers."
  :type 'symbol
  :group 'gpt)

(add-to-list 'savehist-additional-variables 'gpt-command-history)

(defun gpt-display-command-history ()
  "Display the `gpt-command-history' in a buffer."
  (interactive)
  (with-current-buffer (get-buffer-create "*GPT Command History*")
    (erase-buffer)
    (insert (mapconcat #'identity gpt-command-history "\n"))
    (switch-to-buffer (current-buffer))))

(defun gpt-clear-command-history ()
  "Clear the `gpt-command-history' list."
  (interactive)
  (setq gpt-command-history nil)
  (message "GPT command history cleared."))

(defun gpt-export-history (file)
  "Export the `gpt-command-history' to FILE."
  (interactive "Export gpt-command-history to file: ")
  (with-temp-file file
    (dolist (cmd gpt-command-history)
      (insert (format "%s\n" cmd)))))

(defun gpt-completing-read-space (prompt collection &optional predicate require-match initial-input hist def inherit-input-method)
  "Read string in minibuffer with completion, treating space literally.

The arguments are the same as for `completing-read', except that
space does not trigger completion or cycling, but inserts a space
character.  PROMPT is the prompt to display, COLLECTION is the
list of possible completions, and the optional arguments PREDICATE
REQUIRE-MATCH INITIAL-INPUT HIST DEF and INHERIT-INPUT-METHOD
have the same meaning as for `completing-read'."
  (let ((minibuffer-local-completion-map
         (let ((map (copy-keymap minibuffer-local-completion-map)))
           (define-key map " " 'self-insert-command)
           map)))
    (completing-read prompt collection predicate require-match initial-input hist def inherit-input-method)))

(defun gpt-read-command ()
  "Read a GPT command from the user with history and completion."
  (let ((cmd (gpt-completing-read-space "Command: " gpt-command-history nil nil nil 'gpt-command-history)))
    (if (string-equal cmd "n/a")
        ""
      (string-trim cmd))))

(defun gpt-run-buffer (buffer)
  "Run GPT command with BUFFER text as input and append output stream to output-buffer."
  (with-current-buffer buffer
    (goto-char (point-max))
    (font-lock-fontify-buffer)
    (let* ((prompt-file (gpt-create-prompt-file buffer))
           (process (gpt-start-process prompt-file buffer))
           (timer (gpt-start-timer process)))
      (gpt-set-process-sentinel process timer prompt-file)
      (message "GPT: Running command...")
      (font-lock-fontify-buffer))))

(defun gtp-get-template (gpt-api-type)
  "Return the command template by GPT-API-TYPE."
  (if (eq gpt-api-type 'writerai)
      "User: %s\n "
    "User: %s\n\nAssistant: "))

(defun gpt-insert-command (command)
  "Insert COMMAND to GPT in chat format into the current buffer."
  (let ((template (gtp-get-template gpt-api-type)))
    (insert (format template command))))

(defun gpt-get-visible-buffers-content ()
  "Get the content, buffer name, and file name (if available) of all currently visible buffers."
  (let ((visible-buffers (mapcar 'window-buffer (window-list)))
        contents)
    (dolist (buffer visible-buffers contents)
      (with-current-buffer buffer
        (push (format "Buffer Name: %s\nFile Name: %s\nContent:\n%s"
                      (buffer-name)
                      (or (buffer-file-name) "N/A")
                      (buffer-substring-no-properties (point-min) (point-max)))
              contents)))
    (mapconcat 'identity (nreverse contents) "\n\n")))

(defun gpt-dwim (&optional all-buffers)
  "Run user-provided GPT command on region or all visible buffers and print output stream.
If called with a prefix argument (i.e., ALL-BUFFERS is non-nil), use all visible buffers as input. Otherwise, use the current region."
  (interactive "P")
  (let* ((initial-buffer (current-buffer))
         (command (gpt-read-command))
         (output-buffer (gpt-create-output-buffer command))
         (input (if all-buffers
                    (gpt-get-visible-buffers-content)
                  (when (use-region-p)
                    (buffer-substring-no-properties (region-beginning) (region-end))))))
    (switch-to-buffer-other-window output-buffer)
    (if (eq gpt-api-type 'writerai)
        (if input
            (insert (format "User:%s" (base64-encode-string (format "%s \n\n```\n%s\n```\n\n" command input))))
          (gpt-insert-command (base64-encode-string command)))
      (when input
        (insert (format "User:\n\n```\n%s\n```\n\n" input)))
      (gpt-insert-command command))
    (gpt-run-buffer output-buffer)
    (when (eq gpt-api-type 'writerai) (erase-buffer)) ;; to get rid of base64 stuff
    ))

(defun gpt-dwim-all-buffers ()
  "Run user-provided GPT command on all visible buffers and print output stream."
  (interactive)
  (gpt-dwim t))

(defun gpt-follow-up ()
  "Run a follow-up GPT command on the output buffer and append the output stream."
  (interactive)
  (unless (eq major-mode 'gpt-mode)
    (user-error "Not in a gpt output buffer"))
  (let ((command (gpt-read-command)))
    (goto-char (point-max))
    (insert "\n\n")
    (gpt-insert-command command)
    (gpt-run-buffer (current-buffer))))

(defvar gpt-generate-buffer-name-instruction "Create a title with a maximum of 50 chars for the chat above. Say only the title, nothing else. No quotes."
  "The instruction given to GPT to generate a buffer name.")

(defun gpt-get-api-key (gpt-api-type)
  "Get the secret key  by GPT-API-TYPE."
  (cond
   ((eq gpt-api-type 'openai) gpt-openai-key)
   ((eq gpt-api-type 'anthropic) gpt-anthropic-key)
   ((eq gpt-api-type 'writerai) gpt-writerai-key)
   (t (error "Unsupported type %s" gpt-api-type)))
  )

(defun gpt-generate-buffer-name ()
  "Update the buffer name by asking GPT to create a title for it."
  (interactive)
  (unless (eq major-mode 'gpt-mode)
    (user-error "Not in a gpt output buffer"))
  (let* ((gpt-buffer (current-buffer))
         (buffer-string (gpt-buffer-string gpt-buffer))
         (prompt (concat buffer-string "\n\nUser: " gpt-generate-buffer-name-instruction)))
    (with-temp-buffer
      (insert prompt)
      (let ((prompt-file (gpt-create-prompt-file (current-buffer)))
            (api-key (gpt-get-api-key gpt-api-type))
            (api-type-str (symbol-name gpt-api-type)))
        (erase-buffer)
        (message "Asking GPT to generate buffer name...")
        (call-process gpt-python-path nil t nil
                      gpt-script-path api-key gpt-model gpt-max-tokens gpt-temperature api-type-str prompt-file)
        (let ((generated-title (string-trim (buffer-string))))
          (with-current-buffer gpt-buffer
            (rename-buffer (gpt-get-output-buffer-name generated-title))))))))


(defun gpt-buffer-string (buffer)
  "Get BUFFER text as string."
  (with-current-buffer buffer
    (buffer-string)))

(defun gpt-create-prompt-file (buffer)
  "Create a temporary file containing the prompt string from BUFFER text."
  (let ((temp-file (make-temp-file "gpt-prompt")))
    (with-temp-file temp-file
      (insert (gpt-buffer-string buffer)))
    (message "GPT: Prompt written to %s" temp-file)
    temp-file))

(defun gpt-start-process (prompt-file output-buffer)
  "Start the GPT process with the given PROMPT-FILE and OUTPUT-BUFFER.
Use `gpt-script-path' as the executable and pass the other arguments as a list."
  (let* ((api-key (gpt-get-api-key gpt-api-type))
         (api-type-str (symbol-name gpt-api-type))
         (process (start-process "gpt-process" output-buffer
                                 gpt-python-path gpt-script-path
                                 api-key gpt-model gpt-max-tokens gpt-temperature
                                 api-type-str prompt-file)))
    process))

(defvar gpt-buffer-counter 0
  "Counter to ensure unique buffer names for GPT output buffers.")

(defvar gpt-buffer-name-length 60
  "Maximum character length of the GPT buffer name title.")

(defun gpt-get-output-buffer-name (command)
  "Get the output buffer name for a given COMMAND."
  (let* ((truncated-command (substring command 0 (min gpt-buffer-name-length (length command))))
         (ellipsis (if (< (length truncated-command) (length command)) "..." "")))
    (concat "*gpt"
            "[" (number-to-string gpt-buffer-counter) "]: "
            truncated-command
            ellipsis
            "*")))

(defun gpt-create-output-buffer (command)
  "Create a buffer to capture the output of the GPT process.
If `gpt-use-named-buffers' is non-nil, create or get a named buffer.
Otherwise, create a temporary buffer. Use the `gpt-mode' for the output buffer."
  (let ((output-buffer
         (if gpt-use-named-buffers
             (let ((buffer (get-buffer-create (gpt-get-output-buffer-name command))))
               (setq gpt-buffer-counter (1+ gpt-buffer-counter))  ; Increment the counter
               buffer)
           (generate-new-buffer (gpt-get-output-buffer-name command)))))
    (with-current-buffer output-buffer
      (gpt-mode))
    output-buffer))

(defun gpt-start-timer (process)
  "Set timer to run every second and print message if PROCESS is still running."
  (run-with-timer 1 1
                  (lambda (timer-object)
                    (when (process-live-p timer-object)
                      (font-lock-fontify-buffer)
                      (message "GPT: Running...")))
                  process))

(defun gpt-set-process-sentinel (process timer prompt-file)
  "Set a function to run when the PROCESS finishes or fails.

Cancel the timer, delete the prompt file, and print a message with the status.

PROCESS is the GPT process object.
TIMER is the timer object that cancels the process after a timeout.
PROMPT-FILE is the temporary file containing the prompt."
  (set-process-sentinel process
                        (lambda (proc status)
                          (when (memq (process-status proc) '(exit signal))
                            (cancel-timer timer)
                            (if (zerop (process-exit-status proc))
                                (progn
                                  (delete-file prompt-file)
                                  (message "GPT: Finished successfully."))
                              (message "GPT: Failed: %s" status))))))

(defface gpt-input-face
  '((t :inherit comint-highlight-prompt))
  "Face for the input of the GPT commands.")

(defface gpt-output-face
  '((t :inherit default))
  "Face for the output of the GPT commands.")

(defvar gpt-font-lock-keywords
  '(("^\\(User:\\|Human:\\s-*\\)\\(.*\\)$"
     (1 '(face nil invisible gpt-prefix))
     (2 'gpt-input-face))
    ("^\\(Assistant:\\s-*\\)\\(.*\\)$"
     (1 '(face nil invisible gpt-prefix))
     (2 'gpt-output-face))
    ("```\\([\0-\377[:nonascii:]]*?\\)```"  ; match code snippets enclosed in backticks
     (1 'font-lock-constant-face))))

(defun gpt-dynamically-define-gpt-mode ()
  "Define `gpt-mode` based on whether markdown-mode is available or not."
  (let ((parent-mode (if (fboundp 'markdown-mode)
                         'markdown-mode
                       'text-mode)))
    (eval
     ;; the `define-derived-mode` macro expects a literal as its first argument
     ;; hence, we can not simply use the `parent-mode` variable
     ;; but need to use a backquoted list and eval it
     `(define-derived-mode gpt-mode ,parent-mode "GPT"
        "A mode for displaying the output of GPT commands."
        (message "GPT mode intialized with parent: %s" ',parent-mode)
        (setq-local word-wrap t)
        (setq-local font-lock-extra-managed-props '(invisible))
        (if (eq ',parent-mode 'markdown-mode)
            (progn
              (setq markdown-fontify-code-blocks-natively t)
              ;; Combine markdown-mode's keywords with custom keywords
              (setq font-lock-defaults
                    (list (append markdown-mode-font-lock-keywords gpt-font-lock-keywords))))
          (progn
            (setq-local font-lock-defaults '(gpt-font-lock-keywords))
            (font-lock-mode 1)
            (font-lock-fontify-buffer))
          )
        (add-to-invisibility-spec 'gpt-prefix)))))
(gpt-dynamically-define-gpt-mode)

(defun gpt-toggle-prefix ()
  "Toggle the visibility of the GPT prefixes."
  (interactive)
  (if (and (listp buffer-invisibility-spec)
           (memq 'gpt-prefix buffer-invisibility-spec))
      (remove-from-invisibility-spec 'gpt-prefix)
    (add-to-invisibility-spec 'gpt-prefix))
  (font-lock-fontify-buffer))

(defun gpt-copy-code-block ()
  "Copy the content of the code block at point to the clipboard."
  (interactive)
  (let* ((start (if (search-backward "\n```" nil t) (point) nil))
         (_ (goto-char (or (+ start 3) (point-min))))
         (end (if (search-forward "\n```" nil t) (point) nil)))
    (when (and start end)
      (let* ((content (buffer-substring-no-properties (+ start 3) (- end 3)))
             (lang-end (string-match "\n" content))
             (code (if lang-end
                       (substring content (+ lang-end 1))
                     content)))
        (kill-new code)
        (message "Code block copied to clipboard.")))))

(defun gpt-writerai-models ()
  "List writerai models."
  (let ((url-request-extra-headers
         `(("Content-Type" . "application/json")
           ("Authorization" . ,(concat "Bearer " gpt-writerai-key)))))
    (with-current-buffer (url-retrieve-synchronously "https://api.writer.com/v1/models")
      (goto-char url-http-end-of-headers)
      (delete-region (point-min) (point))
      (save-excursion
        (let ((json-object-type 'plist)
              (json-array-type 'list))
          (goto-char (point-min))
          (json-read)))))
  )

(defun gpt-writerai-cache-and-format-models ()
  "Cache and return writerai MODELS."
  (if gpt-writerai-model-cache
      gpt-writerai-model-cache
    (setq gpt-writerai-model-cache
          (mapcar
           (lambda (model)
             (cons
              (plist-get model :name)
              (cons 'writerai (plist-get model :id))))
           (plist-get (gpt-writerai-models) :models)))))

(defun gpt-switch-model ()
  "Switch between models."
  (interactive)
  (let* ((models (append '(("GPT-4o" . (openai . "gpt-4o"))
                           ("Claude 3.5 Sonnet" . (anthropic . "claude-3-5-sonnet-20240620")))
                         (gpt-writerai-cache-and-format-models)))
         (choice (completing-read "Choose model: " (mapcar #'car models) nil t))
         (model-info (cdr (assoc choice models))))
    (setq gpt-api-type (car model-info)
          gpt-model (cdr model-info))
    (message "Switched to %s model: %s" (car model-info) (cdr model-info))))

(define-key gpt-mode-map (kbd "C-c C-c") 'gpt-follow-up)
(define-key gpt-mode-map (kbd "C-c C-p") 'gpt-toggle-prefix)
(define-key gpt-mode-map (kbd "C-c C-b") 'gpt-copy-code-block)
(define-key gpt-mode-map (kbd "C-c C-m") 'gpt-switch-model)
(define-key gpt-mode-map (kbd "C-c C-t") 'gpt-generate-buffer-name)

(provide 'gpt)

;;; gpt.el ends here
