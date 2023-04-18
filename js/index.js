const cv = require("@techstark/opencv-js")
const ort = require("onnxruntime-web")


// for registering state proxy callbacks
const callbacks = {}
// internal state of the app
const state = new Proxy({}, {
    // intercept property set
    set: (obj, key, value) => {
        // change property only if there's a difference
        if (obj[key] == value) return true
        // log and change property
        console.log(`state.${key} = ${obj[key]} => ${value}`)
        obj[key] = value
        // invoke callbacks and return
        try {
            if (key in callbacks) callbacks[key].forEach(callback => callback(value))
        } catch (exc) {
            window.alert(`Not all combinations of options will work, some will fail, like this one - ${exc.toString()}`)
            throw exc
        }
        return true
    }
})


// manual change dispatch
function change(el) {
    // simply dispatch with new event
    el.dispatchEvent(new Event("change"))
}


// plot histogram
function plot_hist(src, dst_el, bins = 256, min = 0, max = 256) {
    // need to pass src as vector
    const srcs = new cv.MatVector()
    srcs.push_back(src)
    // need to pass, not used
    const mask = new cv.Mat()
    // result histogram
    const hist = new cv.Mat()
    // result plot
    const dst = new cv.Mat.zeros(src.rows, bins, cv.CV_8UC3)
    try {
        // grab channels and calculate height of each histogram
        const channels = src.channels()
        const height = Math.trunc(src.rows / channels)
        // start with red and loop channels
        let color = new cv.Scalar(255, 0, 0)
        let channel = -1
        while (++channel < channels) {
            // calculate channel histogram and its bounds
            cv.calcHist(srcs, [channel], mask, hist, [bins], [min, max])
            const bounds = cv.minMaxLoc(hist, mask)
            // calculate histogram plot floor height
            const floor = height * channel + height
            // loop bins
            let bin = -1
            while (++bin < bins) {
                // calculate plot value from hist
                // note that hist contains raw pixel counts, as such normalize with max and scale to height
                const value = Math.trunc(hist.data32F[bin] / bounds.maxVal * (height - 1))
                // don't draw points with no value
                if (!value) continue
                // draw a vertical line from bot to top
                const bot = new cv.Point(bin, floor)
                const top = new cv.Point(bin, floor - value)
                cv.line(dst, bot, top, color, 1)
            }
            // rotate color
            color = new cv.Scalar(color[2], color[0], color[1])
        }
        // show histogram
        cv.imshow(dst_el, dst)
    } finally {
        // deallocate all
        dst.delete()
        hist.delete()
        mask.delete()
        srcs.delete()
    }
}





// stage 1 - original input





// video playback rate
function load_playback_rate(input_playback_rate_el) {
    // simply check and set on internal state
    if (input_playback_rate_el.reportValidity()) state.playback_rate = parseFloat(input_playback_rate_el.value)
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
    // stop browser from default and check if its files or urls
    ev.preventDefault()
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


// original histogram
function load_hist(input_hist_el) {
    // simply check and set on internal state
    if (input_hist_el.reportValidity()) state.hist = input_hist_el.checked
}
function set_hist(hist, output_hist_el) {
    // hide based on value and reload if histogram is required
    output_hist_el.hidden = !hist
    if (hist && state.load) state.load()
}





// stage 2 - initial processing





// initial input
function load_load(load) {
    // store and execute load function
    state.load = load
    load()
}
function load_img(src_el, dst_el, canvas_el, canvas_ctx, width, height, hist_el, initial_hist_el) {
    // coalesce width and height
    canvas_el.width = state.width
    canvas_el.width ||= width
    canvas_el.height = state.height
    canvas_el.height ||= height
    // resize and read new
    canvas_ctx.drawImage(src_el, 0, 0, canvas_el.width, canvas_el.height)
    const mat_initial = cv.imread(canvas_el)
    try {
        // plot original histogram and convert color space
        if (state.hist) plot_hist(mat_initial, hist_el)
        if (state.color_space) cv.cvtColor(mat_initial, mat_initial, cv[state.color_space])
        // show converted and plot histogram
        cv.imshow(dst_el, mat_initial)
        if (state.initial_hist) plot_hist(mat_initial, initial_hist_el)
    } catch (exc) {
        // failed, delete new
        mat_initial.delete()
        throw exc
    }
    // delete previous and set new
    if (state.mat_initial) state.mat_initial.delete()
    state.mat_initial = mat_initial
}


// width
function load_width(input_width_el) {
    // simply check and set on internal state
    if (input_width_el.reportValidity()) state.width = parseInt(input_width_el.value)
}
function set_width() {
    // simply reload
    if (state.load) state.load()
}


// height
function load_height(input_height_el) {
    // simply check and set on internal state
    if (input_height_el.reportValidity()) state.height = parseInt(input_height_el.value)
}
function set_height() {
    // simply reload
    if (state.load) state.load()
}


// color space
function load_color_space(input_color_space_el) {
    // simply check and set on internal state
    if (input_color_space_el.reportValidity()) state.color_space = input_color_space_el.value
}
function set_color_space() {
    // simply reload
    if (state.load) state.load()
}


// video fps
function load_fps(input_fps_el) {
    // simply check and set on internal state
    if (input_fps_el.reportValidity()) state.fps = parseInt(input_fps_el.value)
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


// histogram
function load_initial_hist(input_initial_hist_el) {
    // simply check and set on internal state
    if (input_initial_hist_el.reportValidity()) state.initial_hist = input_initial_hist_el.checked
}
function set_initial_hist(initial_hist, output_initial_hist_el) {
    // hide based on value and reload if histogram is required
    output_initial_hist_el.hidden = !initial_hist
    if (initial_hist && state.load) state.load()
}





// stage 3 - blur





// blur image
function blur_img(dst_el, hist_el) {
    // check if we have previous stage
    if (!state.mat_initial) return
    // allocate dst if we blur, else clone
    const mat_blur = state.blur ? new cv.Mat() : state.mat_initial.clone()
    try {
        // blur based on type
        switch (state.blur) {
            case "blur":
                cv.blur(state.mat_initial, mat_blur, new cv.Size(state.blur_kernel_width, state.blur_kernel_height))
                break
            case "GaussianBlur":
                cv.GaussianBlur(state.mat_initial, mat_blur, new cv.Size(state.blur_kernel_width, state.blur_kernel_height), state.sigma_x, state.sigma_y)
                break
            case "medianBlur":
                cv.medianBlur(state.mat_initial, mat_blur, state.blur_kernel_size)
                break
            case "bilateralFilter":
                cv.bilateralFilter(state.mat_initial, mat_blur, state.blur_diameter, state.sigma_color, state.sigma_space)
                break
        }
        // show blured and plot histogram
        cv.imshow(dst_el, mat_blur)
        if (state.blur_hist) plot_hist(mat_blur, hist_el)
    } catch (exc) {
        // failed, delete blured
        mat_blur.delete()
        throw exc
    }
    // delete previous and set new
    if (state.mat_blur) state.mat_blur.delete()
    state.mat_blur = mat_blur
}


// blur kernel width
function load_blur_kernel_width(input_blur_kernel_width_el) {
    // simply check and set on internal state
    if (input_blur_kernel_width_el.reportValidity()) state.blur_kernel_width = parseInt(input_blur_kernel_width_el.value)
}


// blur kernel height
function load_blur_kernel_height(input_blur_kernel_height_el) {
    // simply check and set on internal state
    if (input_blur_kernel_height_el.reportValidity()) state.blur_kernel_height = parseInt(input_blur_kernel_height_el.value)
}


// blur kernel size
function load_blur_kernel_size(input_blur_kernel_size_el) {
    // simply check and set on internal state
    if (input_blur_kernel_size_el.reportValidity()) state.blur_kernel_size = parseInt(input_blur_kernel_size_el.value)
}


// blur diameter
function load_blur_diameter(input_blur_diameter_el) {
    // simply check and set on internal state
    if (input_blur_diameter_el.reportValidity()) state.blur_diameter = parseInt(input_blur_diameter_el.value)
}


// sigma x
function load_sigma_x(input_sigma_x_el) {
    // simply check and set on internal state
    if (input_sigma_x_el.reportValidity()) state.sigma_x = parseFloat(input_sigma_x_el.value)
}


// sigma y
function load_sigma_y(input_sigma_y_el) {
    // simply check and set on internal state
    if (input_sigma_y_el.reportValidity()) state.sigma_y = parseFloat(input_sigma_y_el.value)
}


// sigma color
function load_sigma_color(input_sigma_color_el) {
    // simply check and set on internal state
    if (input_sigma_color_el.reportValidity()) state.sigma_color = parseInt(input_sigma_color_el.value)
}


// sigma space
function load_sigma_space(input_sigma_space_el) {
    // simply check and set on internal state
    if (input_sigma_space_el.reportValidity()) state.sigma_space = parseInt(input_sigma_space_el.value)
}


// histogram
function load_blur_hist(input_blur_hist_el) {
    // simply check and set on internal state
    if (input_blur_hist_el.reportValidity()) state.blur_hist = input_blur_hist_el.checked
}
function set_blur_hist(blur_hist, output_blur_el, output_blur_hist_el) {
    // hide based on value and reload if histogram is required
    output_blur_hist_el.hidden = !blur_hist
    if (blur_hist) blur_img(output_blur_el, output_blur_hist_el)
}


// blur type
function load_blur(input_blur_el) {
    // simply check and set on internal state
    if (input_blur_el.reportValidity()) state.blur = input_blur_el.value
}
function set_blur(blur, output_blur_el, output_blur_hist_el, input_blur_kernel_width_el, input_blur_kernel_height_el, input_blur_kernel_size_el, input_blur_diameter_el, input_sigma_x_el, input_sigma_y_el, input_sigma_color_el, input_sigma_space_el) {
    input_blur_kernel_width_el.disabled = true
    input_blur_kernel_height_el.disabled = true
    input_blur_kernel_size_el.disabled = true
    input_blur_diameter_el.disabled = true
    input_sigma_x_el.disabled = true
    input_sigma_y_el.disabled = true
    input_sigma_color_el.disabled = true
    input_sigma_space_el.disabled = true
    switch (blur) {
        case "blur":
            input_blur_kernel_width_el.disabled = false
            input_blur_kernel_height_el.disabled = false
            break
        case "GaussianBlur":
            input_blur_kernel_width_el.disabled = false
            input_blur_kernel_height_el.disabled = false
            input_sigma_x_el.disabled = false
            input_sigma_y_el.disabled = false
            break
        case "medianBlur":
            input_blur_kernel_size_el.disabled = false
            break
        case "bilateralFilter":
            input_blur_diameter_el.disabled = false
            input_sigma_color_el.disabled = false
            input_sigma_space_el.disabled = false
            break
    }
    blur_img(output_blur_el, output_blur_hist_el)
}





// stage 4 - threshold





// threshold image
function threshold_img(dst_el, hist_el) {
    // check if we have previous stage
    if (!state.mat_blur) return
    // allocate dst if we threshold, else clone
    const mat_threshold = state.threshold ? new cv.Mat() : state.mat_blur.clone()
    try {
        if (state.threshold) {
            if (state.adaptive_threshold) {
                cv.adaptiveThreshold(state.mat_blur, mat_threshold, state.threshold_max, cv[state.adaptive_threshold], cv[state.threshold], state.threshold_block, state.threshold_constant)
            } else if (state.optimal_threshold) {
                cv.threshold(state.mat_blur, mat_threshold, state.threshold_value, state.threshold_max, cv[state.threshold] | cv[state.optimal_threshold])
            } else {
                cv.threshold(state.mat_blur, mat_threshold, state.threshold_value, state.threshold_max, cv[state.threshold])
            }
        }
        // show thresholded and plot histogram
        cv.imshow(dst_el, mat_threshold)
        if (state.threshold_hist) plot_hist(mat_threshold, hist_el)
    } catch (exc) {
        // failed, delete thresholded
        mat_threshold.delete()
        throw exc
    }
    // delete previous and set new
    if (state.mat_threshold) state.mat_threshold.delete()
    state.mat_threshold = mat_threshold
}


// value
function load_threshold_value(input_threshold_value_el) {
    // simply check and set on internal state
    if (input_threshold_value_el.reportValidity()) state.threshold_value = parseInt(input_threshold_value_el.value)
}


// max
function load_threshold_max(input_threshold_max_el) {
    // simply check and set on internal state
    if (input_threshold_max_el.reportValidity()) state.threshold_max = parseInt(input_threshold_max_el.value)
}


// block
function load_threshold_block(input_threshold_block_el) {
    // simply check and set on internal state
    if (input_threshold_block_el.reportValidity()) state.threshold_block = parseInt(input_threshold_block_el.value)
}


// constant
function load_threshold_constant(input_threshold_constant_el) {
    // simply check and set on internal state
    if (input_threshold_constant_el.reportValidity()) state.threshold_constant = parseInt(input_threshold_constant_el.value)
}


// histogram
function load_threshold_hist(input_threshold_hist_el) {
    // simply check and set on internal state
    if (input_threshold_hist_el.reportValidity()) state.threshold_hist = input_threshold_hist_el.checked
}
function set_threshold_hist(threshold_hist, output_threshold_el, output_threshold_hist_el) {
    // hide based on value and reload if histogram is required
    output_threshold_hist_el.hidden = !threshold_hist
    if (threshold_hist) threshold_img(output_threshold_el, output_threshold_hist_el)
}


// threshold type
function load_threshold(input_threshold_el) {
    // simply check and set on internal state
    if (input_threshold_el.reportValidity()) state.threshold = input_threshold_el.value
}
function set_threshold(threshold, output_threshold_el, output_threshold_hist_el, input_threshold_value_el, input_threshold_max_el, input_optimal_threshold_el, input_adaptive_threshold_el, input_threshold_block_el, input_threshold_constant_el) {
    input_threshold_value_el.disabled = true
    input_threshold_max_el.disabled = true
    input_optimal_threshold_el.disabled = true
    input_adaptive_threshold_el.disabled = true
    input_threshold_block_el.disabled = true
    input_threshold_constant_el.disabled = true
    switch (threshold) {
        case "THRESH_BINARY":
        case "THRESH_BINARY_INV":
            input_threshold_value_el.disabled = false
            input_threshold_max_el.disabled = false
            break
        case "THRESH_TRUNC":
        case "THRESH_TOZERO":
        case "THRESH_TOZERO_INV":
            input_threshold_value_el.disabled = false
            break
        default:
            threshold_img(output_threshold_el, output_threshold_hist_el)
            return
    }
    switch (state.optimal_threshold) {
        case "THRESH_OTSU":
        case "THRESH_TRIANGLE":
            input_threshold_value_el.disabled = true
            break
        default:
            input_adaptive_threshold_el.disabled = false
            break
    }
    switch (state.adaptive_threshold) {
        case "ADAPTIVE_THRESH_MEAN_C":
        case "ADAPTIVE_THRESH_GAUSSIAN_C":
            input_threshold_value_el.disabled = true
            input_threshold_block_el.disabled = false
            input_threshold_constant_el.disabled = false
            break
        default:
            input_optimal_threshold_el.disabled = false
            break
    }
    threshold_img(output_threshold_el, output_threshold_hist_el)
}


// optimal threshold type
function load_optimal_threshold(input_optimal_threshold_el) {
    // simply check and set on internal state
    if (input_optimal_threshold_el.reportValidity()) state.optimal_threshold = input_optimal_threshold_el.value
}


// adaptive threshold type
function load_adaptive_threshold(input_adaptive_threshold_el) {
    // simply check and set on internal state
    if (input_adaptive_threshold_el.reportValidity()) state.adaptive_threshold = input_adaptive_threshold_el.value
}





// initialization





// main entry point
function main() {
    //
    // stage 1 - original input
    //
    // grab original src els
    const img_el = document.getElementById("img")
    const video_el = document.getElementById("video")
    // grab video playback rate el
    const input_playback_rate_el = document.getElementById("input-playback-rate")
    // register callback to react on internal state change
    callbacks.playback_rate = [playback_rate => set_playback_rate(playback_rate, video_el)]
    // register callback to change internal state on el change
    input_playback_rate_el.onchange = () => load_playback_rate(input_playback_rate_el)
    // fire el change to initialize and sync internal state
    change(input_playback_rate_el)
    // same steps for video loop
    const input_loop_el = document.getElementById("input-loop")
    callbacks.loop = [loop => set_loop(loop, video_el)]
    input_loop_el.onchange = () => load_loop(input_loop_el)
    change(input_loop_el)
    // react to img/video src change fired by multiple methods
    callbacks.img_src = [img_src => set_img_src(img_src, img_el, video_el, input_playback_rate_el, input_loop_el)]
    callbacks.video_src = [video_src => set_video_src(video_src, img_el, video_el, input_playback_rate_el, input_loop_el)]
    // same steps for src input from file
    const input_file_el = document.getElementById("input-file")
    input_file_el.onchange = () => load_file(input_file_el)
    change(input_file_el)
    // same steps for src input from url
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
    // react to drag & drop on the entire document
    document.ondragover = load_drag
    document.ondrop = ev => load_drop(ev, input_file_el, input_url_el, submit_fetch_el)
    // grab original histogram el
    const output_hist_el = document.getElementById("output-hist")
    // same steps for histogram input
    const input_hist_el = document.getElementById("input-hist")
    callbacks.hist = [hist => set_hist(hist, output_hist_el)]
    input_hist_el.onchange = () => load_hist(input_hist_el)
    change(input_hist_el)
    //
    // stage 2 - initial processing
    //
    // grab stage output els
    const output_initial_el = document.getElementById("output-initial")
    const output_initial_hist_el = document.getElementById("output-initial-hist")
    // grab temp canvas and its context
    const canvas_el = document.getElementById("canvas")
    const canvas_ctx = canvas_el.getContext("2d")
    // react to original src load
    // note that we remember which one we did last so that we can repeat it when stage config changes
    img_el.onload = () => load_load(() => load_img(
        img_el, output_initial_el,
        canvas_el, canvas_ctx,
        img_el.width, img_el.height,
        output_hist_el, output_initial_hist_el))
    video_el.onloadeddata = () => load_load(() => load_img(
        video_el, output_initial_el,
        canvas_el, canvas_ctx,
        video_el.clientWidth, video_el.clientHeight,
        output_hist_el, output_initial_hist_el))
    // also react to the user seeking to a new time in video
    video_el.onseeked = () => load_img(
        video_el, output_initial_el,
        canvas_el, canvas_ctx,
        video_el.clientWidth, video_el.clientHeight,
        output_hist_el, output_initial_hist_el)
    // same steps for width
    const input_width_el = document.getElementById("input-width")
    callbacks.width = [set_width]
    input_width_el.onchange = () => load_width(input_width_el)
    change(input_width_el)
    // same steps for height
    const input_height_el = document.getElementById("input-height")
    callbacks.height = [set_height]
    input_height_el.onchange = () => load_height(input_height_el)
    change(input_height_el)
    // same steps for color space, but with dynamic options based on whats available on cv
    const input_color_space_el = document.getElementById("input-color-space")
    Object.keys(cv).filter(key => key.startsWith("COLOR_")).forEach(key => {
        const input_color_space_option_el = document.createElement("option")
        input_color_space_option_el.value = key
        input_color_space_option_el.innerHTML = key.slice(6)
        input_color_space_el.appendChild(input_color_space_option_el)
    })
    callbacks.color_space = [set_color_space]
    input_color_space_el.onchange = () => load_color_space(input_color_space_el)
    change(input_color_space_el)
    // same steps for fps, but enable only for video
    // note that this is monitered in set_play
    const input_fps_el = document.getElementById("input-fps")
    callbacks.img_src.push(() => input_fps_el.disabled = true)
    callbacks.video_src.push(() => input_fps_el.disabled = false)
    input_fps_el.onchange = () => load_fps(input_fps_el)
    change(input_fps_el)
    // same steps for video play
    callbacks.play = [set_play]
    video_el.onplay = () => load_play(video_el)
    video_el.onpause = () => load_play(video_el)
    load_play(video_el)
    // same steps for histogram input
    const input_initial_hist_el = document.getElementById("input-initial-hist")
    callbacks.initial_hist = [initial_hist => set_initial_hist(initial_hist, output_initial_hist_el)]
    input_initial_hist_el.onchange = () => load_initial_hist(input_initial_hist_el)
    change(input_initial_hist_el)
    //
    // stage 3 - blur
    //
    // grab stage output els
    const output_blur_el = document.getElementById("output-blur")
    const output_blur_hist_el = document.getElementById("output-blur-hist")
    // react to stage end
    callbacks.mat_initial = [() => blur_img(output_blur_el, output_blur_hist_el)]
    // same steps for blur kernel width input
    const input_blur_kernel_width_el = document.getElementById("input-blur-kernel-width")
    callbacks.blur_kernel_width = [() => blur_img(output_blur_el, output_blur_hist_el)]
    input_blur_kernel_width_el.onchange = () => load_blur_kernel_width(input_blur_kernel_width_el)
    change(input_blur_kernel_width_el)
    // same steps for blur kernel height input
    const input_blur_kernel_height_el = document.getElementById("input-blur-kernel-height")
    callbacks.blur_kernel_height = [() => blur_img(output_blur_el, output_blur_hist_el)]
    input_blur_kernel_height_el.onchange = () => load_blur_kernel_height(input_blur_kernel_height_el)
    change(input_blur_kernel_height_el)
    // same steps for blur kernel size input
    const input_blur_kernel_size_el = document.getElementById("input-blur-kernel-size")
    callbacks.blur_kernel_size = [() => blur_img(output_blur_el, output_blur_hist_el)]
    input_blur_kernel_size_el.onchange = () => load_blur_kernel_size(input_blur_kernel_size_el)
    change(input_blur_kernel_size_el)
    // same steps for blur diameter input
    const input_blur_diameter_el = document.getElementById("input-blur-diameter")
    callbacks.blur_diameter = [() => blur_img(output_blur_el, output_blur_hist_el)]
    input_blur_diameter_el.onchange = () => load_blur_diameter(input_blur_diameter_el)
    change(input_blur_diameter_el)
    // same steps for sigma x input
    const input_sigma_x_el = document.getElementById("input-sigma-x")
    callbacks.sigma_x = [() => blur_img(output_blur_el, output_blur_hist_el)]
    input_sigma_x_el.onchange = () => load_sigma_x(input_sigma_x_el)
    change(input_sigma_x_el)
    // same steps for sigma y input
    const input_sigma_y_el = document.getElementById("input-sigma-y")
    callbacks.sigma_y = [() => blur_img(output_blur_el, output_blur_hist_el)]
    input_sigma_y_el.onchange = () => load_sigma_y(input_sigma_y_el)
    change(input_sigma_y_el)
    // same steps for sigma color input
    const input_sigma_color_el = document.getElementById("input-sigma-color")
    callbacks.sigma_color = [() => blur_img(output_blur_el, output_blur_hist_el)]
    input_sigma_color_el.onchange = () => load_sigma_color(input_sigma_color_el)
    change(input_sigma_color_el)
    // same steps for sigma space input
    const input_sigma_space_el = document.getElementById("input-sigma-space")
    callbacks.sigma_space = [() => blur_img(output_blur_el, output_blur_hist_el)]
    input_sigma_space_el.onchange = () => load_sigma_space(input_sigma_space_el)
    change(input_sigma_space_el)
    // same steps for histogram input
    const input_blur_hist_el = document.getElementById("input-blur-hist")
    callbacks.blur_hist = [blur_hist => set_blur_hist(blur_hist, output_blur_el, output_blur_hist_el)]
    input_blur_hist_el.onchange = () => load_blur_hist(input_blur_hist_el)
    change(input_blur_hist_el)
    // same steps for blur input
    const input_blur_el = document.getElementById("input-blur")
    callbacks.blur = [blur => set_blur(blur, output_blur_el, output_blur_hist_el, input_blur_kernel_width_el, input_blur_kernel_height_el, input_blur_kernel_size_el, input_blur_diameter_el, input_sigma_x_el, input_sigma_y_el, input_sigma_color_el, input_sigma_space_el)]
    input_blur_el.onchange = () => load_blur(input_blur_el)
    change(input_blur_el)
    //
    // stage 4 - threshold
    //
    // grab stage output els
    const output_threshold_el = document.getElementById("output-threshold")
    const output_threshold_hist_el = document.getElementById("output-threshold-hist")
    // react to stage end
    callbacks.mat_blur = [() => threshold_img(output_threshold_el, output_threshold_hist_el)]
    // same steps for value input
    const input_threshold_value_el = document.getElementById("input-threshold-value")
    callbacks.threshold_value = [() => threshold_img(output_threshold_el, output_threshold_hist_el)]
    input_threshold_value_el.onchange = () => load_threshold_value(input_threshold_value_el)
    change(input_threshold_value_el)
    // same steps for max input
    const input_threshold_max_el = document.getElementById("input-threshold-max")
    callbacks.threshold_max = [() => threshold_img(output_threshold_el, output_threshold_hist_el)]
    input_threshold_max_el.onchange = () => load_threshold_max(input_threshold_max_el)
    change(input_threshold_max_el)
    // same steps for block input
    const input_threshold_block_el = document.getElementById("input-threshold-block")
    callbacks.threshold_block = [() => threshold_img(output_threshold_el, output_threshold_hist_el)]
    input_threshold_block_el.onchange = () => load_threshold_block(input_threshold_block_el)
    change(input_threshold_block_el)
    // same steps for constant input
    const input_threshold_constant_el = document.getElementById("input-threshold-constant")
    callbacks.threshold_constant = [() => threshold_img(output_threshold_el, output_threshold_hist_el)]
    input_threshold_constant_el.onchange = () => load_threshold_constant(input_threshold_constant_el)
    change(input_threshold_constant_el)
    // same steps for histogram input
    const input_threshold_hist_el = document.getElementById("input-threshold-hist")
    callbacks.threshold_hist = [threshold_hist => set_threshold_hist(threshold_hist, output_threshold_el, output_threshold_hist_el)]
    input_threshold_hist_el.onchange = () => load_threshold_hist(input_threshold_hist_el)
    change(input_threshold_hist_el)
    // same steps for threshold input
    const input_threshold_el = document.getElementById("input-threshold")
    const input_optimal_threshold_el = document.getElementById("input-optimal-threshold")
    const input_adaptive_threshold_el = document.getElementById("input-adaptive-threshold")
    callbacks.threshold = [threshold => set_threshold(threshold, output_threshold_el, output_threshold_hist_el, input_threshold_value_el, input_threshold_max_el, input_optimal_threshold_el, input_adaptive_threshold_el, input_threshold_block_el, input_threshold_constant_el)]
    callbacks.optimal_threshold = [() => set_threshold(state.threshold, output_threshold_el, output_threshold_hist_el, input_threshold_value_el, input_threshold_max_el, input_optimal_threshold_el, input_adaptive_threshold_el, input_threshold_block_el, input_threshold_constant_el)]
    callbacks.adaptive_threshold = [() => set_threshold(state.threshold, output_threshold_el, output_threshold_hist_el, input_threshold_value_el, input_threshold_max_el, input_optimal_threshold_el, input_adaptive_threshold_el, input_threshold_block_el, input_threshold_constant_el)]
    input_threshold_el.onchange = () => load_threshold(input_threshold_el)
    input_optimal_threshold_el.onchange = () => load_optimal_threshold(input_optimal_threshold_el)
    input_adaptive_threshold_el.onchange = () => load_adaptive_threshold(input_adaptive_threshold_el)
    change(input_threshold_el)
    change(input_optimal_threshold_el)
    change(input_adaptive_threshold_el)
}


// seems that opencv does dynamic initialization, register main to be executed after that
cv.onRuntimeInitialized = main
