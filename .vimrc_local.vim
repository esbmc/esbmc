" The plugin 'local_vimrc.vim' is required for this to be used.
" also, set g_local_vimrc to '.vimrc_local.vim' in .vimrc

" Two space indentation for esbmc, no tabs:

set cindent       " Keep cindent on.
set expandtab     " Use softtabstop
set shiftwidth=2
set softtabstop=2 " Two spaces created by pressing tab

" Load symbol index
set tags=./.ctags;

" Enable clang complete, for completion unsuprisingly
let g:clang_auto_select=1 " Auto-select first popup item, without inserting
let g:clang_hl_errors=1 " Highlight errors

" Don't show scratch/preview window
set completeopt=menu,menuone,longest

function! LoadCscope()
  let db = findfile(".cscope", ".;")
  if (!empty(db))
    let path = strpart(db, 0, match(db, "/.cscope$"))
    set nocscopeverbose " suppress 'duplicate connection' error
    exe "cs add " . db . " " . path
    set cscopeverbose
  endif
endfunction
au VimEnter /* call LoadCscope()
au BufEnter /* call LoadCscope()
