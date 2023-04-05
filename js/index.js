const cv = require("@techstark/opencv-js")
const ort = require("onnxruntime-web")


// for registering state proxy callbacks
const callbacks = {}
// internal state of the app
const state = new Proxy({}, {
    // intercept property set
    set: (obj, key, value) => {
        // log and change property
        console.log(`state.${key} = ${obj[key]} => ${value}`)
        obj[key] = value
        // invoke callbacks and return
        if (key in callbacks) callbacks[key].forEach(callback => callback(value))
        return true
    }
})


// manual change dispatch
function change(el) {
    // simply dispatch with new event
    el.dispatchEvent(new Event("change"))
}


// video playback rate
function load_playback_rate(input_playback_rate_el) {
    // simply check and set on internal state
    if (input_playback_rate_el.reportValidity()) state.playback_rate = input_playback_rate_el.value
}
function set_playback_rate(playback_rate, video_el) {
    // simply set on el
    video_el.playbackRate = playback_rate
}


// video loop
function load_loop(input_loop_el) {
    // simply check and set on internal state
    if (input_loop_el.reportValidity()) state.loop = input_loop_el.checked
}
function set_loop(loop, video_el) {
    // simply set on el
    video_el.loop = loop
}


// img/video src
function load_src(src, type) {
    // src is already prepared, simply check type and set on internal state
    if (type.startsWith("image")) state.img_src = URL.createObjectURL(src)
    else if (type.startsWith("video")) state.video_src = URL.createObjectURL(src)
}
function set_img_src(img_src, img_el, video_el, input_playback_rate_el, input_loop_el) {
    // got img, pause and hide video el
    video_el.pause()
    video_el.hidden = true
    // disable video specific controls
    input_playback_rate_el.disabled = true
    input_loop_el.disabled = true
    // set and show img el
    img_el.src = img_src
    img_el.hidden = false
}
function set_video_src(video_src, img_el, video_el, input_playback_rate_el, input_loop_el) {
    // got video, hide img el
    img_el.hidden = true
    // enable video specific controls
    input_playback_rate_el.disabled = false
    input_loop_el.disabled = false
    // set and show video el
    video_el.src = video_src
    video_el.hidden = false
    // seems that we also need to reset video specific options
    set_playback_rate(state.playback_rate, video_el)
    set_loop(state.loop, video_el)
}


// src input
function load_file(input_file_el) {
    // check el and files and load the first
    if (input_file_el.reportValidity() && input_file_el.files.length) load_src(input_file_el.files[0], input_file_el.files[0].type)
}
async function load_url(input_url_el, proxies) {
    // check el
    if (!input_url_el.reportValidity()) return
    // valid, grab and fetch url
    const url = input_url_el.value
    let response
    try {
        response = await fetch(url)
    } catch {
        // fetch fail, assume cors
        // shuffle proxies for load balance and loop
        proxies.sort(() => .5 - Math.random())
        let proxy_index = 0
        while (proxy_index < proxies.length) {
            try {
                // fetch from proxy
                response = await fetch(`${proxies[proxy_index++]}${url}`)
                // success, stop
                break
            } catch {
                // fail, next proxy
                continue
            }
        }
    }
    // check if we got response and load blob if so
    if (response) load_src(await response.blob(), response.headers.get("Content-Type"))
}


// drop files and urls
function load_drag(ev) {
    // simply stop browser from default
    ev.preventDefault()
}
function load_drop(ev, input_file_el, input_url_el, submit_fetch_el) {
    // stop browser from default and log drop
    ev.preventDefault()
    console.log(ev)
    // check if its files or urls
    if (ev.dataTransfer.files.length) {
        // files, load through file el
        input_file_el.files = ev.dataTransfer.files
        // can't change value, need to fire change event manually
        change(input_file_el)
    } else if (ev.dataTransfer.items.length) ev.dataTransfer.items[0].getAsString(url => {
        // urls, load through url el
        input_url_el.value = url
        submit_fetch_el.click()
    })
}


// processing input
function load_load(load) {
    // store and execute load function
    state.load = load
    load()
}
function load_img(src_el, canvas_el, canvas_ctx, width, height) {
    // coalesce width and height
    canvas_el.width = state.width
    canvas_el.width ||= width
    canvas_el.height = state.height
    canvas_el.height ||= height
    // resize, delete previous and read new
    canvas_ctx.drawImage(src_el, 0, 0, canvas_el.width, canvas_el.height)
    if (state.mat_src) state.mat_src.delete()
    state.mat_src = cv.imread(canvas_el)
}
function set_mat_src(mat_src) {
    // TODO
    console.log(mat_src)
}


// processing width
function load_width(input_width_el) {
    // simply check and set on internal state
    if (input_width_el.reportValidity()) state.width = input_width_el.value
}
function set_width() {
    // simply reload
    if (state.load) state.load()
}


// processing height
function load_height(input_height_el) {
    // simply check and set on internal state
    if (input_height_el.reportValidity()) state.height = input_height_el.value
}
function set_height() {
    // simply reload
    if (state.load) state.load()
}


// processing fps
function load_fps(input_fps_el) {
    // simply check and set on internal state
    if (input_fps_el.reportValidity()) state.fps = input_fps_el.value
}


// video play
function load_play(video_el) {
    // simply set on internal state
    state.play = !video_el.paused && !video_el.ended
}
function set_play(play) {
    if (!play || state.playing) return
    // need to play and not playing, define and schedule callback
    function callback() {
        try {
            // mark start time
            const start = performance.now()
            // reload
            state.load()
            // calculate free time until next frame
            const free = state.fps > 0 ? 1000 / state.fps - (performance.now() - start) : 0
            // schedule next callback if we're still playing
            state.playing = state.play ? setTimeout(callback, Math.max(0, free)) : undefined
        } catch (exc) {
            // exc, stop playing and rethrow
            state.playing = undefined
            throw exc
        }
    }
    state.playing = setTimeout(callback)
}


// main entry point
function main() {
    // original src
    const img_el = document.getElementById("img")
    const video_el = document.getElementById("video")
    // video playback rate
    const input_playback_rate_el = document.getElementById("input-playback-rate")
    callbacks.playback_rate = [playback_rate => set_playback_rate(playback_rate, video_el)]
    input_playback_rate_el.onchange = () => load_playback_rate(input_playback_rate_el)
    change(input_playback_rate_el)
    // video loop
    const input_loop_el = document.getElementById("input-loop")
    callbacks.loop = [loop => set_loop(loop, video_el)]
    input_loop_el.onchange = () => load_loop(input_loop_el)
    change(input_loop_el)
    // img/video src
    callbacks.img_src = [img_src => set_img_src(img_src, img_el, video_el, input_playback_rate_el, input_loop_el)]
    callbacks.video_src = [video_src => set_video_src(video_src, img_el, video_el, input_playback_rate_el, input_loop_el)]
    // src input from file
    const input_file_el = document.getElementById("input-file")
    input_file_el.onchange = () => load_file(input_file_el)
    change(input_file_el)
    // src input from url
    const submit_fetch_el = document.getElementById("submit-fetch")
    const input_url_el = document.getElementById("input-url")
    // proxies in case of cors problems
    const proxies = [
        // "https://cors-proxy.htmldriven.com/?url=",
        "https://corsproxy.io/?",
        // "https://crossorigin.me/",
        // "https://api.allorigins.win/raw?url=",
    ]
    submit_fetch_el.onclick = () => load_url(input_url_el, proxies)
    submit_fetch_el.click()
    // drop
    document.ondragover = load_drag
    document.ondrop = ev => load_drop(ev, input_file_el, input_url_el, submit_fetch_el)
    // processing input
    const canvas_el = document.getElementById("canvas")
    const canvas_ctx = canvas_el.getContext("2d")
    callbacks.mat_src = [set_mat_src]
    img_el.onload = () => load_load(() => load_img(img_el, canvas_el, canvas_ctx, img_el.width, img_el.height))
    video_el.onloadeddata = () => load_load(() => load_img(video_el, canvas_el, canvas_ctx, video_el.clientWidth, video_el.clientHeight))
    video_el.onseeked = () => load_img(video_el, canvas_el, canvas_ctx, video_el.clientWidth, video_el.clientHeight)
    // processing width
    const input_width_el = document.getElementById("input-width")
    callbacks.width = [set_width]
    input_width_el.onchange = () => load_width(input_width_el)
    change(input_width_el)
    // processing height
    const input_height_el = document.getElementById("input-height")
    callbacks.height = [set_height]
    input_height_el.onchange = () => load_height(input_height_el)
    change(input_height_el)
    // processing fps
    const input_fps_el = document.getElementById("input-fps")
    // enable only for video
    callbacks.img_src.push(() => input_fps_el.disabled = true)
    callbacks.video_src.push(() => input_fps_el.disabled = false)
    input_fps_el.onchange = () => load_fps(input_fps_el)
    change(input_fps_el)
    // video play
    callbacks.play = [set_play]
    video_el.onplay = () => load_play(video_el)
    video_el.onpause = () => load_play(video_el)
    load_play(video_el)
}


// seems that opencv does dynamic initialization, register main to be executed after that
cv.onRuntimeInitialized = main
